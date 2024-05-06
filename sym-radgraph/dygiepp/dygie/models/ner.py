import logging
from typing import Any, Dict, List, Optional, Callable

import torch
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from dygie.training.ner_metrics import NERMetrics
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    """
    Named entity recognition module of DyGIE model.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 make_feedforward: Callable,
                 span_emb_dim: int,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._namespaces = [entry for entry in vocab.get_namespaces() if "ner_labels" in entry]

        # Number of classes determine the output dimension of the final layer
        self._n_labels = {name: vocab.get_vocab_size(name) for name in self._namespaces}

        # Null label is needed to keep track of when calculating the metrics
        for namespace in self._namespaces:
            null_label = vocab.get_token_index("", namespace)
            assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        # The output dim is 1 less than the number of labels because we don't score the null label;
        # we just give it a score of 0 by default.

        # Create a separate scorer and metric for each dataset we're dealing with.
        self._ner_scorers = torch.nn.ModuleDict()
        self._ner_metrics = {}

        for namespace in self._namespaces:
            mention_feedforward = make_feedforward(input_dim=span_emb_dim)
            self._ner_scorers[namespace] = torch.nn.Sequential(
                TimeDistributed(mention_feedforward),
                TimeDistributed(torch.nn.Linear(
                    mention_feedforward.get_output_dim(),
                    self._n_labels[namespace] - 1)))

            self._ner_metrics[namespace] = NERMetrics(self._n_labels[namespace], null_label)

        self._active_namespace = None

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings

        self._active_namespace = f"{metadata.dataset}__ner_labels"
        if self._active_namespace not in self._ner_scorers:
            return {"loss": 0}

        scorer = self._ner_scorers[self._active_namespace]

        ner_scores = scorer(span_embeddings)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        ner_scores = util.replace_masked_values(ner_scores, mask.bool(), -1e20)
        # The dummy_scores are the score for the null label.
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        _, predicted_ner = ner_scores.max(2)

        predictions = self.predict(ner_scores,
                                   spans,
                                   span_mask,
                                   metadata)
        output_dict = {"predictions": predictions}

        # if ner_labels is not None:
        #     metrics = self._ner_metrics[self._active_namespace]
        #     metrics(predicted_ner, ner_labels, span_mask)
        #     ner_scores_flat = ner_scores.view(-1, self._n_labels[self._active_namespace])
        #     ner_labels_flat = ner_labels.view(-1)
        #     mask_flat = span_mask.view(-1).bool()

        #     loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])

        #     output_dict["loss"] = loss

        return output_dict

    # updated the predict method to return probabilities
    def predict(self, ner_scores, spans, span_mask, metadata):
        predictions = []
        zipped = zip(ner_scores, spans, span_mask, metadata)
        for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
            softmax_scores = F.softmax(ner_scores_sent, dim=1)  # Compute probabilities
            ix = span_mask_sent.bool()  # Ensure we only consider non-masked spans
    
            predictions_sent = []
            for span, softmax_score in zip(spans_sent[ix], softmax_scores[ix]):
                span_start, span_end = span.tolist()
                label_probs = softmax_score  # Convert softmax scores to a list for all labels
                # Find the predicted label and its probability
                predicted_label_idx = softmax_score.argmax()
                if predicted_label_idx.item() > 0: 
                    predicted_label = self.vocab.get_token_from_index(predicted_label_idx.item(), self._active_namespace)
                    predicted_label_prob = softmax_score[predicted_label_idx]
        
                    prediction = {
                        "span_start": span_start,
                        "span_end": span_end,
                        "predicted_label": predicted_label,
                        "predicted_label_probability": predicted_label_prob,
                        "all_label_probabilities": label_probs
                    }
                    predictions_sent.append(prediction)
    
            predictions.append(predictions_sent)
    
        return predictions


    # TODO(dwadden) This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._ner_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {f"{prefix}_precision": precision,
                         f"{prefix}_recall": recall,
                         f"{prefix}_f1": f1}
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__ner_{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res
