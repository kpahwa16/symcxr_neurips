import json
import os
import networkx as nx
import random
import copy
import itertools
import spacy

from utils import gen_isomorphic_graph, construct_graph
nlp = spacy.load('en_core_web_lg') 

def calculate_subtree_size_helper(graph, root):
    # is leaf node
    if graph.out_degree[root] == 0:
        graph.nodes[root]['subtree_size'] = 1
        return 1

    graph.nodes[root]['subtree_size'] = -1

    children = graph[root]
    total_subtree_size = 1 # root itself
    for child in children:
        if not 'subtree_size' in graph.nodes[child]:
            total_subtree_size += calculate_subtree_size_helper(graph, child)
        
    graph.nodes[root]['subtree_size'] = total_subtree_size
    return total_subtree_size

def calculate_subtree_size(graph):
    # roots = [n for n,d in graph.in_degree() if d==0] 
    for node in graph.nodes():
        if not 'subtree_size' in graph.nodes[node]:
            calculate_subtree_size_helper(graph, node)

def structure_match_helper(G1, G2, t1, t2, g1tog2, g2tog1):

    if not t2 in g1tog2[t1]:
        return False, ([('replace', G1.nodes[t1], G2.nodes[t2], [t1], [t2])], G1.nodes[t1]['subtree_size'] + G2.nodes[t2]['subtree_size']), {t1: (G1.nodes[t1]['subtree_size'] + G2.nodes[t2]['subtree_size'], [('replace', [t1], [t2])],t2)}
    
    # leaf nodes comparison
    if G1.out_degree[t1] == 0 and G2.out_degree[t2] == 0:
        return True, ([], 0), {t1: (0, [], t2)}

    # Only one of them is leaf node:
    if G1.out_degree[t1] > 0 and G2.out_degree[t2] == 0:
        edits = []
        for n1 in nx.descendants(G1, t1):
            edits.append(('add', G1.nodes[n1], [n1], []))
        return False, (edits, G1.nodes[t1]['subtree_size']), {t1: (G1.nodes[t1]['subtree_size'], edits, t2)}
    
    if G1.out_degree[t1] == 0 and G2.out_degree[t2] > 0:
        edits = []
        for n2 in nx.descendants(G2, t2):
            edits.append(('delete', G2.nodes[n2], [], [n2]))
        return False, (edits, G2.nodes[t2]['subtree_size']), {t1:(G2.nodes[t2]['subtree_size'], edits, t2)}

    node_change, node_dist = g1tog2[t1][t2]
    # if node_dist > 0:
    #     print('here')

    # Both are trees
    children1 = G1[t1]
    children2 = G2[t2]

    structure_same_childs = {}

    is_same = (node_dist == 0)
    all_changes = [node_change]
    total_dist = node_dist

    for child1 in children1:
        rel_type_1 = G1[t1][child1]
        if not child1 in g1tog2:
            continue
        possible_child2_ls = g1tog2[child1]
        min_distance = G1.size()

        # Check whether two child trees are the same
        for possible_child2 in possible_child2_ls:
            if possible_child2 in children2:
                rela_difference = []
                child_dist = 0
                rel_type_2 = G2[t2][possible_child2]
                
                if rel_type_1 != rel_type_2:
                    rela_difference = ['replace rela', t1, t2]
                    child_dist = 1

                subtree_is_same, (difference, distance), same_child_mapping = structure_match_helper(G1, G2, child1, possible_child2, g1tog2, g2tog1)
                if min_distance >= distance: 
                    if not child1 in structure_same_childs:
                        structure_same_childs[child1] = []
                    structure_same_childs[child1].append((child_dist + distance, rela_difference + difference, same_child_mapping, possible_child2))
                    min_distance = child_dist + distance

        # If no match child
        if not child1 in structure_same_childs:
            is_same = False
            all_changes.append(('remove', [child1], []))
            total_dist += G1.nodes[child1]['subtree_size']
            continue
        
        # clean the same child with too big difference
        clean_same_child = []
        for dis, difference, cm, child in structure_same_childs[child1]:
            if dis == min_distance:
                clean_same_child.append((dis, difference, cm, child))
        structure_same_childs[child1] = clean_same_child

    # Assert all of the structurely same child only have one correspondance for right now.
    mapped_children2 = []
    all_structure_same_mappings = None

    _, _, new_mapping = preprocess_mapping(G1, G2, structure_same_childs, children1, children2)
    maps = get_all_maps(G1, G2, new_mapping, children1, children2)
    
    min_bijection = False
    min_bijection_dist = G1.size() + G2.size()
    min_changes = []
    min_map = None

    while True:
        try:
            m = next(maps)
        except StopIteration:
            break

        is_bijection, changes, dist = check_bijection(G1 = G1, G2 = G2, mapping=m, from_space=children1, to_space=children2, singleton_mapping={})
        # if dist == 1:
        #     print('here')
        
        if dist < min_bijection_dist: 
            min_bijection = is_bijection
            min_changes = changes
            min_bijection_dist = dist
            min_map = m

    same_child_mapping = {}
    all_changes += min_changes
    total_dist += min_bijection_dist
    is_same &= min_bijection

    for t1, (dist, change, cm, t2) in min_map.items():
        same_child_mapping.update(cm)
        same_child_mapping.update({t1: (total_dist, all_changes, t2)})

    return is_same, (all_changes, total_dist), same_child_mapping

# @timeout(5)  
def compare_graphs(G1, G2):

    calculate_subtree_size(G1)
    calculate_subtree_size(G2)

    g1tog2 = {}
    g2tog1 = {}
    no_matches = []

    # Check the basic equvalency of token and labels
    for n1 in G1.nodes():
        no_match = True
        for n2 in G2.nodes():

            if G1.nodes[n1]['tokens'] == G2.nodes[n2]['tokens'] and \
               G1.nodes[n1]['label'] == G2.nodes[n2]['label']:
                if not n1 in g1tog2:
                    g1tog2[n1] = {}
                if not n2 in g1tog2[n1]:
                    g1tog2[n1][n2] = ([], 0)
                if not n2 in g2tog1:
                    g2tog1[n2] = {}
                if not n1 in g2tog1[n2]:
                    g2tog1[n2][n1] = ([], 0)
                no_match = False

            if G1.nodes[n1]['tokens'] == G2.nodes[n2]['tokens'] and \
               G1.nodes[n1]['label'] != G2.nodes[n2]['label']:
                if not n1 in g1tog2:
                    g1tog2[n1] = {}
                if not n2 in g1tog2[n1]:
                    g1tog2[n1][n2] = (['replace_label', n1, n2], 1)
                if not n2 in g2tog1:
                    g2tog1[n2] = {}
                if not n1 in g2tog1[n2]:
                    g2tog1[n2][n1] = (['replace_label', n2, n1], 1)
                # no_match = False
            
        if no_match:
            no_matches.append(n1)

    # if not len(no_matches) == 0:
    #     return False
        
    # Check the structure to verify the rest
    roots1 = [n for n,d in G1.in_degree() if d==0] 
    roots2 = [n for n,d in G2.in_degree() if d==0] 

    sm_g1tog2 = {}
    sm_g2tog1 = {}

    root_match = {}
    total_difference = []
    total_dist = 0

    # Compare each tree in the forest
    for root1 in roots1:

        # Check each possible match 
        min_distance = G1.size() + G2.size()

        if not root1 in g1tog2:
            # total_difference.append(('delete', root1, []))
            # total_dist += G1.nodes[root1]['subtree_size']
            continue

        for root2 in g1tog2[root1]:
            if root1 == '13':
                print('here')
                
            is_same, diff, same_child_mapping = structure_match_helper(G1, G2, root1, root2, g1tog2, g2tog1)
            if min_distance >= diff[-1]:
                if not root1 in root_match:
                    root_match[root1] = []
                root_match[root1].append((diff[-1], diff, same_child_mapping, root2))
                min_distance = diff[-1]
 
        # If no match root
        if root1 not in root_match:
            # total_difference.append(('delete', root1, []))
            # total_dist += G1.nodes[root1]['subtree_size']
            continue

        clean_root_match = []
        for dist, diff, cm, root in root_match[root1]:
            if dist == min_distance:
                clean_root_match.append((dist, diff, cm, root))
        root_match[root1] = clean_root_match
        
    _, _, new_mapping = preprocess_mapping(G1, G2, root_match, roots1, roots2)
    all_single_maps = get_all_maps(G1, G2, mapping=new_mapping, from_space=roots1, to_space=roots2)

    # obtaining singletons
    child_mapping = {} 
    singleton_mapping = {}
    for gt_n, possible_mapping in new_mapping.items():
        for (d, c, m, pred_n) in possible_mapping:
            child_mapping.update(m)
            child_mapping[gt_n] = (d, c, pred_n)

    for g1_node in g1tog2:
        min_dist = 10000
        min_bijection = False
        min_changes = []
        min_map = None

        if not g1_node in child_mapping:
            singleton_match = []
            for g2_node in g1tog2[g1_node]:
                is_same, (changes, dist), same_child_mapping = structure_match_helper(G1, G2, g1_node, g2_node, g1tog2, g2tog1)
                singleton_mapping.update(same_child_mapping)
                if dist < min_dist:
                    min_dist = dist
                    min_bijection = is_same
                    min_changes = changes
                    min_map = same_child_mapping
                    singleton_match.append((dist, diff, g2_node))

            for dist, diff, g2_node in singleton_match:
                if dist == min_dist:
                    clean_singleton_match = (dist, diff, g2_node)
                    break

            singleton_mapping[g1_node] = clean_singleton_match
            

    is_bijection = False
    min_bijection = False
    min_bijection_dist = 10000
    min_changes = []
    min_map = None

    while True:
        try:
            m = next(all_single_maps)
        except StopIteration:
            break

        # All the children that are used to compare the mapping
        
        is_bijection, changes, dist = check_bijection(G1 = G1, G2 = G2, mapping=m, from_space=roots1, to_space=roots2, singleton_mapping=singleton_mapping)
        # if dist == 1:
        #     print('here')
        
        if dist < min_bijection_dist: 
            min_bijection = is_bijection
            min_changes = changes
            min_bijection_dist = dist
            min_map = m

        # if is_bijection:
        #     break
    
    total_difference += min_changes
    total_dist += min_bijection_dist

    if not min_map is None:
        min_map.update(singleton_mapping)

    new_edits = post_process_difference(G1, G2, total_difference, min_map, g1tog2, g2tog1)
    return min_bijection, new_edits, total_dist

def find_mapping(G1, G2, add_node, mapping, g1tog2, g2tog1, threshold=0.5):
    # first check node token level similarity
    n = G1.nodes[add_node]

    if add_node in g1tog2:
        candidates = g1tog2[add_node]
    else:
        candidates = []
        text = nlp(n['tokens']) 

        for n2 in G2.nodes:
            pred_token = nlp(G2.nodes[n2]['tokens'])
            similarity = text.similarity(pred_token)
            if similarity > threshold:
                candidates.append(n2)

    # second check children level similarity
    
    for candidate in candidates:
        is_same, (changes, dist), same_child_mapping = structure_match_helper(G1, G2, add_node, candidate, g1tog2, g2tog1)
        print('here')

    pass

def post_process_difference(G1, G2, total_difference, mapping, g1tog2, g2tog1):
    new_edits = []
    for edit, node_info, id1_ls, id2_ls in total_difference:
        if edit == 'add':
            assert len(id2_ls) == 0
            for id1 in id1_ls:
                # n1 = find_mapping(G1, G2, id1, mapping, g1tog2, g2tog1, document1, document2, )
                # TODO: need a heuristic here to map the node back
                if not id1 in mapping:
                    continue
                new_edits.append((edit, node_info, None, mapping[id1][-1]))
        elif edit == 'add_rela':
            assert len(id1_ls) == 1 and len(id2_ls) == 1
            id1, id2 = id1_ls[0], id2_ls[0]
            if not id1 in mapping or not id2 in mapping:
                continue
            new_edits.append((edit,  node_info, mapping[id1][-1], mapping[id2][-1]))
        elif edit == 'delete':
            assert len(id1_ls) == 0
            for id2 in id2_ls:
                new_edits.append((edit, node_info, id2, None))
        elif edit == 'delete_rela':
            assert len(id1_ls) == 1 and len(id2_ls) == 1
            id1, id2 = id1_ls[0], id2_ls[0]
            new_edits.append((edit,  node_info, id1, id2))
        else: 
            raise Exception(f'unseen edit: {edit}')  
        
    return new_edits

def check_bijection(G1, G2, mapping, from_space, to_space, singleton_mapping):
    child_mapping = {} 
    g2tog1 = {}
    child_mapping.update(singleton_mapping)

    for gt_n, (d, c, m, pred_n) in mapping.items():
        child_mapping.update(m)
        child_mapping[gt_n] = (d, c, pred_n)

    for gt_n, (d, c, pred_n)  in child_mapping.items():
        g2tog1[pred_n] = gt_n

    to_space = set(to_space)
    all_changes = []
    total_dist = 0
    is_valid = True
    all_tos = set()
    for e1 in from_space:

        if not e1 in mapping:
            is_valid = False
            all_changes.append(('add', G1.nodes[e1], [e1], []))
            total_dist += G1.nodes[e1]['subtree_size']
        else:
            e2 = mapping[e1][-1]
            if e2 is None:
                continue
            if not e2 in to_space:
                is_valid = False
                # Case 1: e2 is a child of one of the node in the to_space
                in_to_space = False

                for n in to_space:
                    nd = nx.descendants(G2, n)
                    nd.add(n)
                    if e2 in nd:
                        in_to_space = True
                        e2d = nx.descendants(G2, e2)
                        e2d.add(e2)
                        to_add_n = set(nd).difference(set(e2d))
                        for diff_node in to_add_n:
                            if not diff_node in g2tog1:
                                all_changes.append(('delete', G2.nodes[diff_node], [], [diff_node]))
                                total_dist += 3
                                continue
                            for to_n in G2[diff_node]:
                                d_n = g2tog1[diff_node]
                                if not to_n in g2tog1:
                                    continue
                                if not g2tog1[to_n] in G1[d_n]:
                                    all_changes.append(('delete_rela', G2[diff_node][to_n], [diff_node], [to_n]))
                                    total_dist += 1 # 2 for token, token label, one for edge label
                    
                # Case 2: no relation between node in to_space
                if not in_to_space:
                    to_delete_n = set(nd).difference(set()).add(n)
                    for diff_node in to_delete_n:
                        if not diff_node in g2tog1:
                            all_changes.append(('add', G1.nodes[e1], [e1], []))
                            total_dist += 2
                        for to_n in G1[diff_node]:
                            d_n = child_mapping[diff_node]
                            if not child_mapping[to_n] in G2[d_n]:
                                all_changes.append(('add_rela', G1[diff_node][to_n], [diff_node], [to_n]))
                                total_dist += 1 # 2 for token, token label, one for edge label

            all_tos.add(e2)
            
    return is_valid, all_changes, total_dist


def preprocess_mapping(G1, G2, mapping, from_space, to_space):
    changes = []
    distance = 0
    new_mapping = {}
    mapped_space = []

    # There are extra 
    for from_node in from_space:
        no_mapping = False

        if not from_node in mapping:
            no_mapping = True
        else:
            new_children = []
            children = mapping[from_node]
            for dis, difference, cm, child in children:
                # if child in to_space:
                new_children.append((dis, difference, cm, child))
                mapped_space.append(child)
            if len(new_children) == 0:
                no_mapping = True
            else:
                new_mapping[from_node] = new_children

    # for to_node in to_space:
    #     if not to_node in mapped_space:
    #         changes.append(('delete', G2.nodes[to_node], [], [to_node]))
            # distance += G2.nodes[to_node]['subtree_size']

    # We need to find the largest non overlap single maps
    # map_from_space = set(new_mapping.keys())
    # map_to_space = set([v[-1] for vs in new_mapping.values() for v in vs])
    # if len(map_to_space) < len(map_from_space):
    #     for from_node in new_mapping:
    #         to_space = new_mapping[from_node]
    #         to_space.append((G1.nodes[from_node]['subtree_size'], ('delete', [from_node], []), {}, None))
        # print('here')
        
    
    # if '43' in new_mapping and '47' in new_mapping:
    #     print('here')

    return distance, changes, new_mapping

def get_all_maps(G1, G2, mapping, from_space, to_space):
    # TODO: rank the mappings heuristically with smallest changes to largest
    # I remember this is none trivial
    single_maps = itertools.product(*mapping.values())

    for single_map in single_maps:
        has_dup = False

        changes = []
        diff = 0

        all_v = set()
        for v in single_map:
            if v[-1] is None:
                continue
            if v[-1] not in all_v:
                all_v.add(v[-1])
            else: 
                has_dup = True
                break

        yield {k: single_map[ct] for ct, k in enumerate(mapping)}


timeouts = [82]
if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../data'))
    pred_dir = os.path.join(data_dir, 'Radgraph')
    gt_dir = os.path.join(data_dir, 'medical_imaging')
    check_split = 'pred'

    if check_split == 'train':  
        gt_data_path = os.path.join(gt_dir, 'train.json')
        pred_data_path = os.path.join(pred_dir, 'train-pred.json')
    else:
        gt_data_path = os.path.join(gt_dir, 'test.json')
        pred_data_path = os.path.join(pred_dir, 'test-pred.json')


    gt_data_points = json.load(open(gt_data_path))
    pred_data_points = json.load(open(pred_data_path))

    gt_graph_reports = {}
    gt_texts = {}
    pred_texts = {}
    all_outcomes = []
    timeout_ct = 0

    for file_name, label_info in gt_data_points.items():
        gt_graph_reports[file_name] = []
        gt_texts[file_name] = label_info['text']
        for tag, report in label_info.items():
            # For each manual labeler, create a parse version of the report
            if 'labeler' in tag:
                G = construct_graph(report)
                gt_graph_reports[file_name].append(G)
            
            elif 'entities' in tag:
                G = construct_graph(label_info)
                gt_graph_reports[file_name].append(G)

    pred_graph_reports = {}
    for file_name, label_info in pred_data_points.items():
        pred_graph_reports[file_name] = []
        pred_texts[file_name] = label_info['text']
        for tag, report in label_info.items():
            # For each manual labeler, create a parse version of the report
            if 'labeler' in tag:
                G = construct_graph(report)
                pred_graph_reports[file_name].append(G)

            elif 'entities' in tag:
                G = construct_graph(label_info)
                pred_graph_reports[file_name].append(G)
    
    ct = 0 
    for file_name, pred_graphs in pred_graph_reports.items():
        # print(file_name)

        gt_graphs = gt_graph_reports[file_name]
        for pred_graph in pred_graphs:
            for gt_graph in gt_graphs:
                
                if ct in timeouts:
                    continue

                # if not ct == 4:
                #     ct += 1
                #     continue

                print(ct)
                print(file_name)
                try:
                    outcome = compare_graphs(gt_graph, pred_graph,  gt_texts[file_name], pred_texts[file_name])
                    all_outcomes.append(outcome)
                    print(outcome)
                except TimeoutError:
                    timeout_ct += 1

                ct += 1
                
    same_ct = len([i for i in all_outcomes if i[0]])
    different_ct =  len([i for i in all_outcomes if not i])

    print(same_ct)
    print(different_ct)
    print(timeout_ct)
    print('end')