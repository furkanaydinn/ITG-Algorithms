use std::collections::HashMap;
use std::cmp::Ordering;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use rayon::prelude::*;

// --- Veri Yapıları ve Yardımcı Fonksiyonlar ---
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct Interval {
    s_time: u32,
    c_time: u32,
    travel_time: u32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct PathInfo {
    arrival_time: u32,
    length: u32,
}

impl Ord for PathInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        self.arrival_time.cmp(&other.arrival_time)
            .then_with(|| self.length.cmp(&other.length))
    }
}

impl PartialOrd for PathInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn manage_dominance_and_sort(paths: &mut Vec<PathInfo>) {
    if paths.is_empty() {
        return;
    }
    paths.sort_unstable();
    paths.dedup(); // Tekrarları kaldır (sıralı olduğu için ardışık olanları)

    let mut non_dominated: Vec<PathInfo> = Vec::new();
    if paths.is_empty() { return; } // dedup sonrası boş kalabilir

    for current_path in paths.iter() {
        let mut is_dominated_by_existing = false;
        for existing_path in non_dominated.iter() {
            if existing_path.arrival_time <= current_path.arrival_time && existing_path.length <= current_path.length {
                is_dominated_by_existing = true;
                break;
            }
        }
        if is_dominated_by_existing {
            continue;
        }
        non_dominated.retain(|existing_path| {
            !(current_path.arrival_time <= existing_path.arrival_time && current_path.length <= existing_path.length)
        });
        non_dominated.push(*current_path);
        non_dominated.sort_unstable(); // Sıralamayı koru
    }
    *paths = non_dominated;
}

type Graph = HashMap<usize, HashMap<usize, Vec<Interval>>>;

fn generate_deterministic_itg_rust(
    num_vertices: usize,
    edge_density: f64,
    max_intervals_per_edge: usize,
    max_time_val: u32,
    max_duration_per_interval: u32,
    max_travel_time: u32,
    seed: u64,
) -> Graph {
    println!("Deterministik ITG üretiliyor (Rust, seed={})...", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut graph: Graph = HashMap::new();
    for u in 0..num_vertices {
        for v in 0..num_vertices {
            if u == v { continue; }
            if rng.r#gen::<f64>() < edge_density {
                let mut intervals_for_edge: Vec<Interval> = Vec::new();
                let num_intervals = rng.gen_range(1..=max_intervals_per_edge);
                let mut current_time = 0;
                for _ in 0..num_intervals {
                    let start_time_offset_divisor = (num_intervals * 2).saturating_add(1) as u32;
                    let start_time_offset_max = if start_time_offset_divisor > 0 { max_time_val / start_time_offset_divisor } else { max_time_val };
                    let start_time_offset = rng.gen_range(1..=std::cmp::max(1, start_time_offset_max));

                    let s_i = current_time + start_time_offset;
                    let duration = rng.gen_range(1..=max_duration_per_interval);
                    let c_i = s_i + duration;
                    let travel_time = rng.gen_range(1..=max_travel_time);
                    if c_i < max_time_val {
                        intervals_for_edge.push(Interval { s_time: s_i, c_time: c_i, travel_time });
                        current_time = c_i;
                    } else {
                        break;
                    }
                }
                if !intervals_for_edge.is_empty() {
                    intervals_for_edge.sort_unstable_by_key(|interval| interval.s_time);
                    graph.entry(u).or_default().insert(v, intervals_for_edge);
                }
            }
        }
    }
    graph
}

// --- Sıralı Algoritma ---
fn itg_shortest_path_algorithm2_rust_sequential(
    graph_adj: &Graph,
    num_vertices: usize,
    start_vertex_id: usize,
) -> Vec<Option<PathInfo>> {
    let mut p_all_hops: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
    let mut p_new_prev_hop: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
    let mut shrtst_p: Vec<Option<PathInfo>> = vec![None; num_vertices];

    if start_vertex_id < num_vertices {
        p_all_hops[start_vertex_id].push(PathInfo { arrival_time: 0, length: 0 });
        p_new_prev_hop[start_vertex_id].push(PathInfo { arrival_time: 0, length: 0 });
        shrtst_p[start_vertex_id] = Some(PathInfo { arrival_time: 0, length: 0 });
    } else {
        eprintln!("Geçersiz başlangıç düğümü ID'si");
        return shrtst_p;
    }

    let mut new_paths_count_in_iteration = 1;
    let mut k = 0;

    while k < num_vertices.saturating_sub(1) && new_paths_count_in_iteration > 0 {
        k += 1;
        new_paths_count_in_iteration = 0;
        
        let mut p_new_candidates_current_hop: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
        for u in 0..num_vertices {
            if p_new_prev_hop[u].is_empty() { continue; }
            if let Some(neighbors) = graph_adj.get(&u) {
                for (v_idx, intervals) in neighbors.iter() {
                     if *v_idx >= num_vertices { continue; }
                    for prev_path_to_u in p_new_prev_hop[u].iter() {
                        for interval in intervals.iter() {
                            let mut departure_time_from_u: Option<u32> = None;
                            if prev_path_to_u.arrival_time <= interval.s_time {
                                departure_time_from_u = Some(interval.s_time);
                            } else if prev_path_to_u.arrival_time <= interval.c_time {
                                departure_time_from_u = Some(prev_path_to_u.arrival_time);
                            }
                            if let Some(dep_time) = departure_time_from_u {
                                p_new_candidates_current_hop[*v_idx].push(PathInfo {
                                    arrival_time: dep_time + interval.travel_time,
                                    length: prev_path_to_u.length + interval.travel_time,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        let mut p_new_current_hop_for_next_iter: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
        for v_node_idx in 0..num_vertices {
            if !p_new_candidates_current_hop[v_node_idx].is_empty() {
                let mut candidates = p_new_candidates_current_hop[v_node_idx].clone();
                manage_dominance_and_sort(&mut candidates);
                p_new_current_hop_for_next_iter[v_node_idx] = candidates;
            }

            // P_all_hops ve shrtst_p güncelle
            if !p_new_current_hop_for_next_iter[v_node_idx].is_empty() || !p_all_hops[v_node_idx].is_empty() {
                let mut merged_paths = p_all_hops[v_node_idx].clone();
                merged_paths.extend_from_slice(&p_new_current_hop_for_next_iter[v_node_idx]);
                manage_dominance_and_sort(&mut merged_paths);
                p_all_hops[v_node_idx] = merged_paths;

                let mut current_best_len = shrtst_p[v_node_idx].map_or(u32::MAX, |p| p.length);
                let mut best_path_opt = shrtst_p[v_node_idx];

                for path in p_all_hops[v_node_idx].iter() {
                    if path.length < current_best_len {
                        best_path_opt = Some(*path);
                        current_best_len = path.length;
                    } else if path.length == current_best_len {
                        if let Some(current_shortest) = best_path_opt {
                            if path.arrival_time < current_shortest.arrival_time {
                                best_path_opt = Some(*path);
                            }
                        } else {
                             best_path_opt = Some(*path);
                        }
                    }
                }
                shrtst_p[v_node_idx] = best_path_opt;
            }
            if !p_new_current_hop_for_next_iter[v_node_idx].is_empty() {
                new_paths_count_in_iteration = 1;
            }
        }
        p_new_prev_hop = p_new_current_hop_for_next_iter;
    }
    shrtst_p
}

// --- Paralel Algoritma (Rayon) ---
fn itg_shortest_path_algorithm2_rust_parallel(
    graph_adj: &Graph,
    num_vertices: usize,
    start_vertex_id: usize,
) -> Vec<Option<PathInfo>> {
    let mut p_all_hops: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
    let mut p_new_prev_hop: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
    // shrtst_p'yi Arc<Mutex<>> ile sarmalamak yerine, sonunda toplayacağız
    let mut shrtst_p_temp: Vec<Option<PathInfo>> = vec![None; num_vertices];


    if start_vertex_id < num_vertices {
        p_all_hops[start_vertex_id].push(PathInfo { arrival_time: 0, length: 0 });
        p_new_prev_hop[start_vertex_id].push(PathInfo { arrival_time: 0, length: 0 });
        shrtst_p_temp[start_vertex_id] = Some(PathInfo { arrival_time: 0, length: 0 });
    } else {
        eprintln!("Geçersiz başlangıç düğümü ID'si");
        return shrtst_p_temp;
    }

    let mut new_paths_count_in_iteration = 1; // Atomik olmasına gerek yok, ana thread kontrol ediyor
    let mut k = 0;

    while k < num_vertices.saturating_sub(1) && new_paths_count_in_iteration > 0 {
        k += 1;
        
        // Adım 1: Sıralı (Daha önce tartışıldığı gibi)
        let mut p_new_candidates_current_hop_sequential: Vec<Vec<PathInfo>> = vec![Vec::new(); num_vertices];
        for u in 0..num_vertices {
            if p_new_prev_hop[u].is_empty() { continue; }
            if let Some(neighbors) = graph_adj.get(&u) {
                for (v_idx, intervals) in neighbors.iter() {
                     if *v_idx >= num_vertices { continue; }
                    for prev_path_to_u in p_new_prev_hop[u].iter() {
                        for interval in intervals.iter() {
                            let mut departure_time_from_u: Option<u32> = None;
                            if prev_path_to_u.arrival_time <= interval.s_time {
                                departure_time_from_u = Some(interval.s_time);
                            } else if prev_path_to_u.arrival_time <= interval.c_time {
                                departure_time_from_u = Some(prev_path_to_u.arrival_time);
                            }
                            if let Some(dep_time) = departure_time_from_u {
                                p_new_candidates_current_hop_sequential[*v_idx].push(PathInfo {
                                    arrival_time: dep_time + interval.travel_time,
                                    length: prev_path_to_u.length + interval.travel_time,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        // Adım 2: P_new(v,k) oluşturma (Paralel)
        let p_new_current_hop_for_next_iter_parallel: Vec<Vec<PathInfo>> = (0..num_vertices)
            .into_par_iter()
            .map(|v_node_idx| {
                if !p_new_candidates_current_hop_sequential[v_node_idx].is_empty() {
                    let mut candidates = p_new_candidates_current_hop_sequential[v_node_idx].clone();
                    manage_dominance_and_sort(&mut candidates);
                    candidates
                } else {
                    Vec::new()
                }
            })
            .collect();

        // Adım 3: P_all_hops ve shrtst_p güncelleme (Paralel)
        // Her v_node için (yeni P_all_hops[v], yeni shrtstP[v], yeni yol bulundu mu?) tuple'ı topla
        let results_for_v: Vec<(Vec<PathInfo>, Option<PathInfo>, bool)> = (0..num_vertices)
            .into_par_iter()
            .map(|v_node_idx| {
                // p_all_hops ve shrtst_p_temp read-only olarak erişilir (önceki iterasyondan)
                // veya bu iterasyonun başında kopyalanır.
                // Burada p_all_hops'ı klonluyoruz, çünkü her thread kendi parçasını güncelleyecek.
                let mut current_p_all_v = p_all_hops[v_node_idx].clone();
                let mut current_shrtst_v = shrtst_p_temp[v_node_idx]; // Önceki shrtstP değeri
                let mut path_found_for_this_v = false;

                if !p_new_current_hop_for_next_iter_parallel[v_node_idx].is_empty() {
                    current_p_all_v.extend_from_slice(&p_new_current_hop_for_next_iter_parallel[v_node_idx]);
                    manage_dominance_and_sort(&mut current_p_all_v);

                    let mut current_best_len = current_shrtst_v.map_or(u32::MAX, |p| p.length);
                    // shrtstP güncelleme
                    for path in current_p_all_v.iter() {
                        if path.length < current_best_len {
                            current_shrtst_v = Some(*path);
                            current_best_len = path.length;
                        } else if path.length == current_best_len {
                            if let Some(current_s) = current_shrtst_v {
                                if path.arrival_time < current_s.arrival_time {
                                    current_shrtst_v = Some(*path);
                                }
                            } else {
                                current_shrtst_v = Some(*path);
                            }
                        }
                    }
                    path_found_for_this_v = true; // Yeni k-hoplu yol işlendi
                }
                (current_p_all_v, current_shrtst_v, path_found_for_this_v)
            })
            .collect();

        new_paths_count_in_iteration = 0; // Sıfırla
        for v_node_idx in 0..num_vertices {
            p_all_hops[v_node_idx] = results_for_v[v_node_idx].0.clone(); // Klonlama burada da gerekli olabilir
            shrtst_p_temp[v_node_idx] = results_for_v[v_node_idx].1;
            if results_for_v[v_node_idx].2 && !p_new_current_hop_for_next_iter_parallel[v_node_idx].is_empty() {
                // Sadece gerçekten P_new_current_hop'ta yeni bir şey varsa iterasyonu devam ettir.
                new_paths_count_in_iteration = 1;
            }
        }
        
        p_new_prev_hop = p_new_current_hop_for_next_iter_parallel;
    }
    shrtst_p_temp
}


// --- Main Bloğu ---
fn main() {
    const BIG_NUM_VERTICES_RUST: usize = 200;
    const EDGE_DENSITY_BIG_RUST: f64 = 0.05;
    const MAX_INTERVALS_BIG_RUST: usize = 2;
    const MAX_TIME_BIG_RUST: u32 = 300;
    const MAX_DURATION_BIG_RUST: u32 = 20;
    const MAX_TRAVEL_BIG_RUST: u32 = 10;
    const SHARED_SEED_RUST: u64 = 123;
    const START_NODE_BIG_RUST: usize = 0;

    println!("--- Rust (Büyük Graf, {} düğüm) ---", BIG_NUM_VERTICES_RUST);
    let itg_big_rust = generate_deterministic_itg_rust(
        BIG_NUM_VERTICES_RUST, EDGE_DENSITY_BIG_RUST, MAX_INTERVALS_BIG_RUST,
        MAX_TIME_BIG_RUST, MAX_DURATION_BIG_RUST, MAX_TRAVEL_BIG_RUST, SHARED_SEED_RUST
    );
    println!("Graf oluşturuldu. Anahtar sayısı: {}, Yaklaşık toplam alt-harita girdisi: {}",
        itg_big_rust.len(),
        itg_big_rust.values().map(|inner_map| inner_map.len()).sum::<usize>()
    );

    // Sıralı Rust
    println!("\nSıralı Rust çalıştırılıyor...");
    let start_rs_seq = Instant::now();
    let shrtstP_rs_seq = itg_shortest_path_algorithm2_rust_sequential(&itg_big_rust, BIG_NUM_VERTICES_RUST, START_NODE_BIG_RUST);
    let duration_rs_seq = start_rs_seq.elapsed();
    println!("Sıralı Rust örnek sonuçları (ilk 3 ve son 2):");
    for i_idx in (0..3).chain(BIG_NUM_VERTICES_RUST-2..BIG_NUM_VERTICES_RUST) {
        if i_idx < BIG_NUM_VERTICES_RUST { // Sınır kontrolü
             println!("  Düğüm {}: {:?}", i_idx, shrtstP_rs_seq.get(i_idx).unwrap_or(&None));
        }
    }
    println!("Sıralı Rust süresi: {:?}", duration_rs_seq);

    // Paralel Rust (Rayon)
    println!("\nParalel Rust (Rayon) çalıştırılıyor...");
    let start_rs_par = Instant::now();
    let shrtstP_rs_par = itg_shortest_path_algorithm2_rust_parallel(&itg_big_rust, BIG_NUM_VERTICES_RUST, START_NODE_BIG_RUST);
    let duration_rs_par = start_rs_par.elapsed();
     println!("Paralel Rust örnek sonuçları (ilk 3 ve son 2):");
    for i_idx in (0..3).chain(BIG_NUM_VERTICES_RUST-2..BIG_NUM_VERTICES_RUST) {
         if i_idx < BIG_NUM_VERTICES_RUST { // Sınır kontrolü
            println!("  Düğüm {}: {:?}", i_idx, shrtstP_rs_par.get(i_idx).unwrap_or(&None));
        }
    }
    println!("Paralel Rust (Rayon) süresi: {:?}", duration_rs_par);
}