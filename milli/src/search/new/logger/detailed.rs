use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::random;
use roaring::RoaringBitmap;

use crate::search::new::query_term::{LocatedQueryTerm, QueryTerm, WordDerivations};
use crate::search::new::ranking_rule_graph::{
    Edge, EdgeCondition, EmptyPathsCache, ProximityGraph, RankingRuleGraph, RankingRuleGraphTrait,
    TypoGraph,
};
use crate::search::new::small_bitmap::SmallBitmap;
use crate::search::new::{QueryGraph, QueryNode, SearchContext};
use crate::search::new::{RankingRule, SearchLogger};

pub enum SearchEvents {
    RankingRuleStartIteration {
        ranking_rule_idx: usize,
        query: QueryGraph,
        universe: RoaringBitmap,
        time: Instant,
    },
    RankingRuleNextBucket {
        ranking_rule_idx: usize,
        universe: RoaringBitmap,
        candidates: RoaringBitmap,
        time: Instant,
    },
    RankingRuleEndIteration {
        ranking_rule_idx: usize,
        universe: RoaringBitmap,
        time: Instant,
    },
    ExtendResults {
        new: Vec<u32>,
    },
    WordsState {
        query_graph: QueryGraph,
    },
    ProximityState {
        graph: RankingRuleGraph<ProximityGraph>,
        paths: Vec<Vec<u16>>,
        empty_paths_cache: EmptyPathsCache,
        universe: RoaringBitmap,
        distances: Vec<Vec<(u16, SmallBitmap)>>,
        cost: u16,
    },
    TypoState {
        graph: RankingRuleGraph<TypoGraph>,
        paths: Vec<Vec<u16>>,
        empty_paths_cache: EmptyPathsCache,
        universe: RoaringBitmap,
        distances: Vec<Vec<(u16, SmallBitmap)>>,
        cost: u16,
    },
    RankingRuleSkipBucket {
        ranking_rule_idx: usize,
        candidates: RoaringBitmap,
        time: Instant,
    },
}

pub struct DetailedSearchLogger {
    folder_path: PathBuf,
    initial_query: Option<QueryGraph>,
    initial_query_time: Option<Instant>,
    query_for_universe: Option<QueryGraph>,
    initial_universe: Option<RoaringBitmap>,
    ranking_rules_ids: Option<Vec<String>>,
    events: Vec<SearchEvents>,
}
impl DetailedSearchLogger {
    pub fn new(folder_path: &str) -> Self {
        Self {
            folder_path: PathBuf::new().join(folder_path),
            initial_query: None,
            initial_query_time: None,
            query_for_universe: None,
            initial_universe: None,
            ranking_rules_ids: None,
            events: vec![],
        }
    }
}

impl SearchLogger<QueryGraph> for DetailedSearchLogger {
    fn initial_query(&mut self, query: &QueryGraph) {
        self.initial_query = Some(query.clone());
        self.initial_query_time = Some(Instant::now());
    }

    fn query_for_universe(&mut self, query: &QueryGraph) {
        self.query_for_universe = Some(query.clone());
    }

    fn initial_universe(&mut self, universe: &RoaringBitmap) {
        self.initial_universe = Some(universe.clone());
    }
    fn ranking_rules(&mut self, rr: &[&mut dyn RankingRule<QueryGraph>]) {
        self.ranking_rules_ids = Some(rr.iter().map(|rr| rr.id()).collect());
    }

    fn start_iteration_ranking_rule<'transaction>(
        &mut self,
        ranking_rule_idx: usize,
        _ranking_rule: &dyn RankingRule<'transaction, QueryGraph>,
        query: &QueryGraph,
        universe: &RoaringBitmap,
    ) {
        self.events.push(SearchEvents::RankingRuleStartIteration {
            ranking_rule_idx,
            query: query.clone(),
            universe: universe.clone(),
            time: Instant::now(),
        })
    }

    fn next_bucket_ranking_rule<'transaction>(
        &mut self,
        ranking_rule_idx: usize,
        _ranking_rule: &dyn RankingRule<'transaction, QueryGraph>,
        universe: &RoaringBitmap,
        candidates: &RoaringBitmap,
    ) {
        self.events.push(SearchEvents::RankingRuleNextBucket {
            ranking_rule_idx,
            universe: universe.clone(),
            candidates: candidates.clone(),
            time: Instant::now(),
        })
    }
    fn skip_bucket_ranking_rule<'transaction>(
        &mut self,
        ranking_rule_idx: usize,
        _ranking_rule: &dyn RankingRule<'transaction, QueryGraph>,
        candidates: &RoaringBitmap,
    ) {
        self.events.push(SearchEvents::RankingRuleSkipBucket {
            ranking_rule_idx,
            candidates: candidates.clone(),
            time: Instant::now(),
        })
    }

    fn end_iteration_ranking_rule<'transaction>(
        &mut self,
        ranking_rule_idx: usize,
        _ranking_rule: &dyn RankingRule<'transaction, QueryGraph>,
        universe: &RoaringBitmap,
    ) {
        self.events.push(SearchEvents::RankingRuleEndIteration {
            ranking_rule_idx,
            universe: universe.clone(),
            time: Instant::now(),
        })
    }
    fn add_to_results(&mut self, docids: &[u32]) {
        self.events.push(SearchEvents::ExtendResults { new: docids.to_vec() });
    }

    fn log_words_state(&mut self, query_graph: &QueryGraph) {
        self.events.push(SearchEvents::WordsState { query_graph: query_graph.clone() });
    }

    fn log_proximity_state(
        &mut self,
        query_graph: &RankingRuleGraph<ProximityGraph>,
        paths_map: &[Vec<u16>],
        empty_paths_cache: &EmptyPathsCache,
        universe: &RoaringBitmap,
        distances: Vec<Vec<(u16, SmallBitmap)>>,
        cost: u16,
    ) {
        self.events.push(SearchEvents::ProximityState {
            graph: query_graph.clone(),
            paths: paths_map.to_vec(),
            empty_paths_cache: empty_paths_cache.clone(),
            universe: universe.clone(),
            distances,
            cost,
        })
    }

    fn log_typo_state(
        &mut self,
        query_graph: &RankingRuleGraph<TypoGraph>,
        paths_map: &[Vec<u16>],
        empty_paths_cache: &EmptyPathsCache,
        universe: &RoaringBitmap,
        distances: Vec<Vec<(u16, SmallBitmap)>>,
        cost: u16,
    ) {
        self.events.push(SearchEvents::TypoState {
            graph: query_graph.clone(),
            paths: paths_map.to_vec(),
            empty_paths_cache: empty_paths_cache.clone(),
            universe: universe.clone(),
            distances,
            cost,
        })
    }
}

impl DetailedSearchLogger {
    pub fn write_d2_description(&self, ctx: &mut SearchContext) {
        let mut prev_time = self.initial_query_time.unwrap();
        let mut timestamp = vec![];
        fn activated_id(timestamp: &[usize]) -> String {
            let mut s = String::new();
            s.push('0');
            for t in timestamp.iter() {
                s.push_str(&format!("{t}"));
            }
            s
        }

        let index_path = self.folder_path.join("index.d2");
        let mut file = std::fs::File::create(index_path).unwrap();
        writeln!(&mut file, "direction: right").unwrap();
        writeln!(&mut file, "Initial Query Graph: {{").unwrap();
        let initial_query_graph = self.initial_query.as_ref().unwrap();
        Self::query_graph_d2_description(ctx, initial_query_graph, &mut file);
        writeln!(&mut file, "}}").unwrap();

        writeln!(&mut file, "Query Graph Used To Compute Universe: {{").unwrap();
        let query_graph_for_universe = self.query_for_universe.as_ref().unwrap();
        Self::query_graph_d2_description(ctx, query_graph_for_universe, &mut file);
        writeln!(&mut file, "}}").unwrap();

        let initial_universe = self.initial_universe.as_ref().unwrap();
        writeln!(&mut file, "Initial Universe Length {}", initial_universe.len()).unwrap();

        writeln!(&mut file, "Control Flow Between Ranking Rules: {{").unwrap();
        writeln!(&mut file, "shape: sequence_diagram").unwrap();
        for (idx, rr_id) in self.ranking_rules_ids.as_ref().unwrap().iter().enumerate() {
            writeln!(&mut file, "{idx}: {rr_id}").unwrap();
        }
        writeln!(&mut file, "results").unwrap();
        // writeln!(&mut file, "time").unwrap();
        for event in self.events.iter() {
            match event {
                SearchEvents::RankingRuleStartIteration { ranking_rule_idx, time, .. } => {
                    let _elapsed = time.duration_since(prev_time);
                    prev_time = *time;
                    let parent_activated_id = activated_id(&timestamp);
                    timestamp.push(0);
                    let self_activated_id = activated_id(&timestamp);
                    // writeln!(&mut file, "time.{self_activated_id}: {:.2}", elapsed.as_micros() as f64 / 1000.0).unwrap();
                    if *ranking_rule_idx != 0 {
                        let parent_ranking_rule_idx = ranking_rule_idx - 1;
                        writeln!(
                            &mut file,
                            "{parent_ranking_rule_idx}.{parent_activated_id} -> {ranking_rule_idx}.{self_activated_id} : start iteration",
                        )
                        .unwrap();
                    }
                    writeln!(
                        &mut file,
                        "{ranking_rule_idx}.{self_activated_id} {{
    style {{
        fill: \"#D8A7B1\"
    }}
}}"
                    )
                    .unwrap();
                }
                SearchEvents::RankingRuleNextBucket {
                    ranking_rule_idx,
                    time,
                    universe,
                    candidates,
                } => {
                    let _elapsed = time.duration_since(prev_time);
                    prev_time = *time;
                    let old_activated_id = activated_id(&timestamp);
                    // writeln!(&mut file, "time.{old_activated_id}: {:.2}", elapsed.as_micros() as f64 / 1000.0).unwrap();
                    *timestamp.last_mut().unwrap() += 1;
                    let next_activated_id = activated_id(&timestamp);
                    writeln!(&mut file,
                        "{ranking_rule_idx}.{old_activated_id} -> {ranking_rule_idx}.{next_activated_id} : next bucket {}/{}", candidates.len(), universe.len())
                        .unwrap();
                }
                SearchEvents::RankingRuleSkipBucket { ranking_rule_idx, candidates, time } => {
                    let _elapsed = time.duration_since(prev_time);
                    prev_time = *time;
                    let old_activated_id = activated_id(&timestamp);
                    // writeln!(&mut file, "time.{old_activated_id}: {:.2}", elapsed.as_micros() as f64 / 1000.0).unwrap();
                    *timestamp.last_mut().unwrap() += 1;
                    let next_activated_id = activated_id(&timestamp);
                    let len = candidates.len();
                    writeln!(&mut file,
                        "{ranking_rule_idx}.{old_activated_id} -> {ranking_rule_idx}.{next_activated_id} : skip bucket ({len})",)
                        .unwrap();
                }
                SearchEvents::RankingRuleEndIteration { ranking_rule_idx, time, .. } => {
                    let _elapsed = time.duration_since(prev_time);
                    prev_time = *time;
                    let cur_activated_id = activated_id(&timestamp);
                    // writeln!(&mut file, "time.{cur_activated_id}: {:.2}", elapsed.as_micros() as f64 / 1000.0).unwrap();

                    timestamp.pop();
                    let parent_activated_id = activated_id(&timestamp);
                    let parent_ranking_rule = if *ranking_rule_idx == 0 {
                        "start".to_owned()
                    } else {
                        format!("{}.{parent_activated_id}", ranking_rule_idx - 1)
                    };
                    writeln!(
                        &mut file,
                        "{ranking_rule_idx}.{cur_activated_id} -> {parent_ranking_rule} : end iteration",
                    )
                    .unwrap();
                }
                SearchEvents::ExtendResults { new } => {
                    if new.is_empty() {
                        continue;
                    }
                    let cur_ranking_rule = timestamp.len() - 1;
                    let cur_activated_id = activated_id(&timestamp);
                    let docids = new.iter().collect::<Vec<_>>();
                    let len = new.len();
                    let random = random::<u64>();

                    writeln!(
                        &mut file,
                        "{cur_ranking_rule}.{cur_activated_id} -> results.{random} : \"add {len}\"
results.{random} {{
    tooltip: \"{docids:?}\"
    style {{
        fill: \"#B6E2D3\"
    }}
}}
"
                    )
                    .unwrap();
                }
                SearchEvents::WordsState { query_graph } => {
                    let cur_ranking_rule = timestamp.len() - 1;
                    *timestamp.last_mut().unwrap() += 1;
                    let cur_activated_id = activated_id(&timestamp);
                    *timestamp.last_mut().unwrap() -= 1;
                    let id = format!("{cur_ranking_rule}.{cur_activated_id}");
                    let new_file_path = self.folder_path.join(format!("{id}.d2"));
                    let mut new_file = std::fs::File::create(new_file_path).unwrap();
                    Self::query_graph_d2_description(ctx, query_graph, &mut new_file);
                    writeln!(
                        &mut file,
                        "{id} {{
    link: \"{id}.d2.svg\"
}}"
                    )
                    .unwrap();
                }
                SearchEvents::ProximityState {
                    graph,
                    paths,
                    empty_paths_cache,
                    universe,
                    distances,
                    cost,
                } => {
                    let cur_ranking_rule = timestamp.len() - 1;
                    *timestamp.last_mut().unwrap() += 1;
                    let cur_activated_id = activated_id(&timestamp);
                    *timestamp.last_mut().unwrap() -= 1;
                    let id = format!("{cur_ranking_rule}.{cur_activated_id}");
                    let new_file_path = self.folder_path.join(format!("{id}.d2"));
                    let mut new_file = std::fs::File::create(new_file_path).unwrap();
                    Self::ranking_rule_graph_d2_description(
                        ctx,
                        graph,
                        paths,
                        empty_paths_cache,
                        distances.clone(),
                        &mut new_file,
                    );
                    writeln!(
                        &mut file,
                        "{id} {{
    link: \"{id}.d2.svg\"
    tooltip: \"cost {cost}, universe len: {}\"
}}",
                        universe.len()
                    )
                    .unwrap();
                }
                SearchEvents::TypoState {
                    graph,
                    paths,
                    empty_paths_cache,
                    universe,
                    distances,
                    cost,
                } => {
                    let cur_ranking_rule = timestamp.len() - 1;
                    *timestamp.last_mut().unwrap() += 1;
                    let cur_activated_id = activated_id(&timestamp);
                    *timestamp.last_mut().unwrap() -= 1;
                    let id = format!("{cur_ranking_rule}.{cur_activated_id}");
                    let new_file_path = self.folder_path.join(format!("{id}.d2"));
                    let mut new_file = std::fs::File::create(new_file_path).unwrap();
                    Self::ranking_rule_graph_d2_description(
                        ctx,
                        graph,
                        paths,
                        empty_paths_cache,
                        distances.clone(),
                        &mut new_file,
                    );
                    writeln!(
                        &mut file,
                        "{id} {{
    link: \"{id}.d2.svg\"
    tooltip: \"cost {cost}, universe len: {}\"
}}",
                        universe.len()
                    )
                    .unwrap();
                }
            }
        }
        writeln!(&mut file, "}}").unwrap();
    }

    fn query_node_d2_desc(
        ctx: &mut SearchContext,
        node_idx: usize,
        node: &QueryNode,
        distances: &[(u16, SmallBitmap)],
        file: &mut File,
    ) {
        match &node {
            QueryNode::Term(LocatedQueryTerm { value, .. }) => match value {
                QueryTerm::Phrase { phrase } => {
                    let phrase = ctx.phrase_interner.get(*phrase);
                    let phrase_str = phrase.description(&ctx.word_interner);
                    writeln!(file, "{node_idx} : \"{phrase_str}\"").unwrap();
                }
                QueryTerm::Word { derivations } => {
                    let WordDerivations {
                        original,
                        zero_typo,
                        one_typo,
                        two_typos,
                        use_prefix_db,
                        synonyms,
                        split_words,
                    } = ctx.derivations_interner.get(*derivations);

                    let original = ctx.word_interner.get(*original);
                    writeln!(
                        file,
                        "{node_idx} : \"{original}\" {{
shape: class"
                    )
                    .unwrap();
                    for w in zero_typo.iter().copied() {
                        let w = ctx.word_interner.get(w);
                        writeln!(file, "\"{w}\" : 0").unwrap();
                    }
                    for w in one_typo.iter().copied() {
                        let w = ctx.word_interner.get(w);
                        writeln!(file, "\"{w}\" : 1").unwrap();
                    }
                    for w in two_typos.iter().copied() {
                        let w = ctx.word_interner.get(w);
                        writeln!(file, "\"{w}\" : 2").unwrap();
                    }
                    if let Some(split_words) = split_words {
                        let phrase = ctx.phrase_interner.get(*split_words);
                        let phrase_str = phrase.description(&ctx.word_interner);
                        writeln!(file, "\"{phrase_str}\" : split_words").unwrap();
                    }
                    for synonym in synonyms.iter().copied() {
                        let phrase = ctx.phrase_interner.get(synonym);
                        let phrase_str = phrase.description(&ctx.word_interner);
                        writeln!(file, "\"{phrase_str}\" : synonym").unwrap();
                    }
                    if *use_prefix_db {
                        writeln!(file, "use prefix DB : true").unwrap();
                    }
                    for (d, edges) in distances.iter() {
                        writeln!(file, "\"distance {d}\" : {:?}", edges.iter().collect::<Vec<_>>())
                            .unwrap();
                    }

                    writeln!(file, "}}").unwrap();
                }
            },
            QueryNode::Deleted => panic!(),
            QueryNode::Start => {
                writeln!(file, "{node_idx} : START").unwrap();
            }
            QueryNode::End => {
                writeln!(file, "{node_idx} : END").unwrap();
            }
        }
    }
    fn query_graph_d2_description(
        ctx: &mut SearchContext,
        query_graph: &QueryGraph,
        file: &mut File,
    ) {
        writeln!(file, "direction: right").unwrap();
        for node in 0..query_graph.nodes.len() {
            if matches!(query_graph.nodes[node], QueryNode::Deleted) {
                continue;
            }
            Self::query_node_d2_desc(ctx, node, &query_graph.nodes[node], &[], file);

            for edge in query_graph.edges[node].successors.iter() {
                writeln!(file, "{node} -> {edge};\n").unwrap();
            }
        }
    }
    fn ranking_rule_graph_d2_description<R: RankingRuleGraphTrait>(
        ctx: &mut SearchContext,
        graph: &RankingRuleGraph<R>,
        paths: &[Vec<u16>],
        _empty_paths_cache: &EmptyPathsCache,
        distances: Vec<Vec<(u16, SmallBitmap)>>,
        file: &mut File,
    ) {
        writeln!(file, "direction: right").unwrap();

        writeln!(file, "Proximity Graph {{").unwrap();
        for (node_idx, node) in graph.query_graph.nodes.iter().enumerate() {
            if matches!(node, QueryNode::Deleted) {
                continue;
            }
            let distances = &distances[node_idx];
            Self::query_node_d2_desc(ctx, node_idx, node, distances.as_slice(), file);
        }
        for edge in graph.edges_store.iter().flatten() {
            let Edge { source_node, dest_node, condition: details, .. } = edge;

            match &details {
                EdgeCondition::Unconditional => {
                    writeln!(
                        file,
                        "{source_node} -> {dest_node} : \"always cost {cost}\"",
                        cost = edge.cost,
                    )
                    .unwrap();
                }
                EdgeCondition::Conditional(details) => {
                    writeln!(
                        file,
                        "{source_node} -> {dest_node} : \"cost {cost} {edge_label}\"",
                        cost = edge.cost,
                        edge_label = R::label_for_edge_condition(details)
                    )
                    .unwrap();
                }
            }
        }
        writeln!(file, "}}").unwrap();

        // writeln!(file, "Distances {{").unwrap();
        // Self::paths_d2_description(graph, paths, file);
        // writeln!(file, "}}").unwrap();

        writeln!(file, "Shortest Paths {{").unwrap();
        Self::paths_d2_description(ctx, graph, paths, file);
        writeln!(file, "}}").unwrap();

        // writeln!(file, "Empty Edge Couples {{").unwrap();
        // for (i, (e1, e2)) in empty_paths_cache.empty_couple_edges.iter().enumerate() {
        //     writeln!(file, "{i} : \"\" {{").unwrap();
        //     Self::edge_d2_description(graph, *e1, file);
        //     Self::edge_d2_description(graph, *e2, file);
        //     writeln!(file, "{e1} -- {e2}").unwrap();
        //     writeln!(file, "}}").unwrap();
        // }
        // writeln!(file, "}}").unwrap();

        // writeln!(file, "Removed Edges {{").unwrap();
        // for edge_idx in empty_paths_cache.empty_edges.iter() {
        //     writeln!(file, "{edge_idx}").unwrap();
        // }
        // writeln!(file, "}}").unwrap();
    }
    fn edge_d2_description<R: RankingRuleGraphTrait>(
        ctx: &mut SearchContext,
        graph: &RankingRuleGraph<R>,
        edge_idx: u16,
        file: &mut File,
    ) {
        let Edge { source_node, dest_node, cost, .. } =
            graph.edges_store[edge_idx as usize].as_ref().unwrap();
        let source_node = &graph.query_graph.nodes[*source_node as usize];
        let source_node_desc = match source_node {
            QueryNode::Term(term) => match term.value {
                QueryTerm::Phrase { phrase } => {
                    let phrase = ctx.phrase_interner.get(phrase);
                    phrase.description(&ctx.word_interner)
                }
                QueryTerm::Word { derivations } => {
                    let derivations = ctx.derivations_interner.get(derivations);
                    ctx.word_interner.get(derivations.original).to_owned()
                }
            },
            QueryNode::Deleted => panic!(),
            QueryNode::Start => "START".to_owned(),
            QueryNode::End => "END".to_owned(),
        };
        let dest_node = &graph.query_graph.nodes[*dest_node as usize];
        let dest_node_desc = match dest_node {
            QueryNode::Term(term) => match term.value {
                QueryTerm::Phrase { phrase } => {
                    let phrase = ctx.phrase_interner.get(phrase);
                    phrase.description(&ctx.word_interner)
                }
                QueryTerm::Word { derivations } => {
                    let derivations = ctx.derivations_interner.get(derivations);
                    ctx.word_interner.get(derivations.original).to_owned()
                }
            },
            QueryNode::Deleted => panic!(),
            QueryNode::Start => "START".to_owned(),
            QueryNode::End => "END".to_owned(),
        };
        writeln!(
            file,
            "{edge_idx}: \"{source_node_desc}->{dest_node_desc} [{cost}]\" {{
            shape: class
        }}"
        )
        .unwrap();
    }
    fn paths_d2_description<R: RankingRuleGraphTrait>(
        ctx: &mut SearchContext,
        graph: &RankingRuleGraph<R>,
        paths: &[Vec<u16>],
        file: &mut File,
    ) {
        for (path_idx, edge_indexes) in paths.iter().enumerate() {
            writeln!(file, "{path_idx} {{").unwrap();
            for edge_idx in edge_indexes.iter() {
                Self::edge_d2_description(ctx, graph, *edge_idx, file);
            }
            for couple_edges in edge_indexes.windows(2) {
                let [src_edge_idx, dest_edge_idx] = couple_edges else { panic!() };
                writeln!(file, "{src_edge_idx} -> {dest_edge_idx}").unwrap();
            }
            writeln!(file, "}}").unwrap();
        }
    }
}