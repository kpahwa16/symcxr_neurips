'''

Modified the forward function
Modified the make_output_human_readable function
'''

import logging
from typing import Dict, List, Optional, Union
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, TimeDistributed
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner import NERTagger
from dygie.models.relation import RelationExtractor
from dygie.models.events import EventExtractor
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie")
class DyGIE(Model):
    """
    TODO(dwadden) document me.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    submodule_params: ``TODO(dwadden)``
        A nested dictionary specifying parameters to be passed on to initialize submodules.
    max_span_width: ``int``
        The maximum width of candidate spans.
    target_task: ``str``:
        The task used to make early stopping decisions.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    module_initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the individual modules.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    display_metrics: ``List[str]``. A list of the metrics that should be printed out during model
        training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 target_task: str,
                 feedforward_params: Dict[str, Union[int, float]],
                 loss_weights: Dict[str, float],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 module_initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None) -> None:
        super(DyGIE, self).__init__(vocab, regularizer)

        ####################

        # Create span extractor.
        self._endpoint_span_extractor = EndpointSpanExtractor(
            embedder.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False)

        ####################

        # Set parameters.
        self._embedder = embedder
        self._loss_weights = loss_weights
        self._max_span_width = max_span_width
        self._display_metrics = self._get_display_metrics(target_task)
        token_emb_dim = self._embedder.get_output_dim()
        span_emb_dim = self._endpoint_span_extractor.get_output_dim()

        ####################

        # Create submodules.

        modules = Params(modules)

        # Helper function to create feedforward networks.
        def make_feedforward(input_dim):
            return FeedForward(input_dim=input_dim,
                               num_layers=feedforward_params["num_layers"],
                               hidden_dims=feedforward_params["hidden_dims"],
                               activations=torch.nn.ReLU(),
                               dropout=feedforward_params["dropout"])

        # Submodules

        self._ner = NERTagger.from_params(vocab=vocab,
                                          make_feedforward=make_feedforward,
                                          span_emb_dim=span_emb_dim,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))

        self._coref = CorefResolver.from_params(vocab=vocab,
                                                make_feedforward=make_feedforward,
                                                span_emb_dim=span_emb_dim,
                                                feature_size=feature_size,
                                                params=modules.pop("coref"))

        self._relation = RelationExtractor.from_params(vocab=vocab,
                                                       make_feedforward=make_feedforward,
                                                       span_emb_dim=span_emb_dim,
                                                       feature_size=feature_size,
                                                       params=modules.pop("relation"))

        self._events = EventExtractor.from_params(vocab=vocab,
                                                  make_feedforward=make_feedforward,
                                                  token_emb_dim=token_emb_dim,
                                                  span_emb_dim=span_emb_dim,
                                                  feature_size=feature_size,
                                                  params=modules.pop("events"))

        ####################

        # Initialize text embedder and all submodules
        for module in [self._ner, self._coref, self._relation, self._events]:
            module_initializer(module)

        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        The `target` is the name of the task used to make early stopping decisions. Show metrics
        related to this task.
        """
        lookup = {
            "ner": [f"MEAN__{name}" for name in
                    ["ner_precision", "ner_recall", "ner_f1"]],
            "relation": [f"MEAN__{name}" for name in
                         ["relation_precision", "relation_recall", "relation_f1"]],
            "coref": ["coref_precision", "coref_recall", "coref_f1", "coref_mention_recall"],
            "events": [f"MEAN__{name}" for name in
                       ["trig_class_f1", "arg_class_f1"]]}
        if target_task not in lookup:
            raise ValueError(f"Invalied value {target_task} has been given as the target task.")
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO(dwadden) Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)


    # modified the forward function
    @overrides
    def forward(self,
                text,
                spans,
                metadata,
                ner_labels=None,
                coref_labels=None,
                relation_labels=None,
                trigger_labels=None,
                argument_labels=None):
        """
        TODO(dwadden) change this.
        """
        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        if relation_labels is not None:
            relation_labels = relation_labels.long()
        if argument_labels is not None:
            argument_labels = argument_labels.long()

        # TODO(dwadden) Multi-document minibatching isn't supported yet. For now, get rid of the
        # extra dimension in the input tensors. Will return to this once the model runs.
        if len(metadata) > 1:
            raise NotImplementedError("Multi-document minibatching not supported.")

        metadata = metadata[0]
        spans = self._debatch(spans)  # (n_sents, max_n_spans, 2)
        ner_labels = self._debatch(ner_labels)  # (n_sents, max_n_spans)
        coref_labels = self._debatch(coref_labels)  #  (n_sents, max_n_spans)
        relation_labels = self._debatch(relation_labels)  # (n_sents, max_n_spans, max_n_spans)
        trigger_labels = self._debatch(trigger_labels)  # TODO(dwadden)
        argument_labels = self._debatch(argument_labels)  # TODO(dwadden)

        # Encode using BERT, then debatch.
        # Since the data are batched, we use `num_wrapping_dims=1` to unwrap the document dimension.
        # (1, n_sents, max_sententence_length, embedding_dim)

        # TODO(dwadden) Deal with the case where the input is longer than 512.
        text_embeddings = self._embedder(text, num_wrapping_dims=1)
        text_embeddings = self._debatch(text_embeddings)
        text_mask = self._debatch(util.get_text_field_mask(text, num_wrapping_dims=1).float())
        sentence_lengths = text_mask.sum(dim=1).long()
        
        span_mask = (spans[:, :, 0] >= 0).float()
        spans = F.relu(spans.float()).long()
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        
        output_dict = {}
        output_dict['metadata'] = metadata

        # NER Module
        if self._loss_weights['ner'] > 0:
            ner_output = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)
            output_dict['ner'] = ner_output
        
        # Coreference Module
        if self._loss_weights['coref'] > 0:
            coref_output, coref_indices = self._coref.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, coref_labels, metadata)
            coref_output = self._coref.coref_propagation(coref_output)
            span_embeddings = self._coref.update_spans(coref_output, span_embeddings, coref_indices)
            output_dict['coref'] = self._coref.predict_labels(coref_output, metadata)
        
        # Relation Module
        if self._loss_weights['relation'] > 0:
            relation_output = self._relation(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)
            output_dict['relation'] = relation_output
        
        # Event Module
        if self._loss_weights['events'] > 0:
            events_output = self._events(
                text_mask, text_embeddings, spans, span_mask, span_embeddings,
                sentence_lengths, trigger_labels, argument_labels, ner_labels, metadata)
            output_dict['events'] = events_output
        
        # Aggregate losses from different modules
        # total_loss = sum(self._loss_weights[task] * output_dict.get(task, {}).get("loss", 0)
        #                  for task in ['ner', 'coref', 'relation', 'events'])
        # output_dict['loss'] = total_loss
    
        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings,
                               top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr,
                                    span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

    # modified the make_output_human_readable function
    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        doc = copy.deepcopy(output_dict["metadata"])
    
        # Process Coreference output for human readability, if applicable.
        if self._loss_weights["coref"] > 0:
            decoded_coref = self._coref.make_output_human_readable(output_dict["coref"])["predicted_clusters"][0]
            sentences = doc.sentences
            sentence_starts = [sent.sentence_start for sent in sentences]
            predicted_clusters = [document.Cluster(entry, i, sentences, sentence_starts)
                                  for i, entry in enumerate(decoded_coref)]
            doc.predicted_clusters = predicted_clusters
    
        # Process NER output for human readability, including probabilities, if applicable.
        if self._loss_weights["ner"] > 0:
            for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
                sentence.predicted_ner = [{
                    "span": (prediction["span_start"], prediction["span_end"]),
                    "label": prediction["predicted_label"],
                    "probability": prediction["predicted_label_probability"]
                } for prediction in predictions]
    
        # Process Relations output for human readability, including probabilities, if applicable.
        if self._loss_weights["relation"] > 0:
            for predictions, sentence in zip(output_dict["relation"]["predictions"], doc):
                sentence.predicted_relations = [{
                    "span1": prediction["span1"],
                    "span2": prediction["span2"],
                    "label": prediction["label"],
                    "probability": prediction["probability"]
                } for prediction in predictions]
    
        # Process Events output for human readability, including probabilities, if applicable.
        if self._loss_weights["events"] > 0:
            for predictions, sentence in zip(output_dict["events"]["predictions"], doc):
                sentence.predicted_events = [{
                    "trigger": (prediction["trigger"]["start"], prediction["trigger"]["end"]),
                    "trigger_label": prediction["trigger"]["label"],
                    "trigger_probability": prediction["trigger"]["probability"],
                    "arguments": [{
                        "span": (arg["span_start"], arg["span_end"]),
                        "label": arg["label"],
                        "probability": arg["probability"]
                    } for arg in prediction["arguments"]]
                } for prediction in predictions]
    
        return doc

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)
        metrics_events = self._events.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_coref.keys()) + list(metrics_ner.keys()) +
                        list(metrics_relation.keys()) + list(metrics_events.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_coref.items()) +
                           list(metrics_ner.items()) +
                           list(metrics_relation.items()) +
                           list(metrics_events.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
