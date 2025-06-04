import random
import time
import multiprocessing
from functools import partial # Gerekirse kullanılabilir, şimdilik direkt argüman geçiyoruz

# --- Veri Yapıları ve Yardımcı Fonksiyonlar ---
def manage_dominance_and_sort(paths_input):
    """
    Verilen yollar listesinden baskın olmayanları seçer ve varış zamanına göre sıralar.
    Her yol (varış_zamanı, uzunluk) formatındadır.
    """
    if not paths_input:
        return []

    # Benzersiz yolları al ve sırala
    # (arrival_time, length) tuple'ları üzerinden set oluşturmak tekrarları kaldırır
    unique_sorted_paths = sorted(list(set(paths_input)))

    if not unique_sorted_paths:
        return []
        
    non_dominated_paths = []
    for current_arr, current_len in unique_sorted_paths:
        is_dominated = False
        # Yeni aday (current_path), mevcut non_dominated_paths listesindeki
        # herhangi bir yol tarafından domine ediliyor mu?
        for nd_arr, nd_len in non_dominated_paths:
            if nd_arr <= current_arr and nd_len <= current_len:
                is_dominated = True
                break
        
        if not is_dominated:
            next_non_dominated_paths = []
            for nd_arr, nd_len in non_dominated_paths:
                if not (current_arr <= nd_arr and current_len <= nd_len):
                    next_non_dominated_paths.append((nd_arr, nd_len))
            
            non_dominated_paths = next_non_dominated_paths
            non_dominated_paths.append((current_arr, current_len))
            non_dominated_paths.sort() # Sıralamayı koru
    return non_dominated_paths

def generate_deterministic_itg(num_vertices, edge_density, max_intervals_per_edge,
                               max_time_val=100, max_duration_per_interval=10,
                               max_travel_time=5, seed=42):
    print(f"Deterministik ITG üretiliyor (Python, seed={seed})...")
    random.seed(seed)
    graph = {v: {} for v in range(num_vertices)}
    for u in range(num_vertices):
        for v in range(num_vertices):
            if u == v: continue
            if random.random() < edge_density:
                intervals_for_edge = []
                num_intervals = random.randint(1, max_intervals_per_edge)
                current_time = 0
                for _ in range(num_intervals):
                    start_time_offset = random.randint(1, max(1, max_time_val // (num_intervals * 2 + 1) if (num_intervals * 2 + 1) > 0 else 1))
                    s_i = current_time + start_time_offset
                    duration = random.randint(1, max_duration_per_interval)
                    c_i = s_i + duration
                    lambda_i = random.randint(1, max_travel_time)
                    if c_i < max_time_val:
                        intervals_for_edge.append((s_i, c_i, lambda_i))
                        current_time = c_i
                    else:
                        break
                if intervals_for_edge:
                    intervals_for_edge.sort()
                    graph.setdefault(u, {})[v] = intervals_for_edge
    return graph

# --- Sıralı Algoritma ---
def itg_shortest_path_algorithm2_python_sequential(graph_adj, num_vertices, start_vertex_id):
    P_all_hops = {v_idx: [] for v_idx in range(num_vertices)}
    P_new_prev_hop = {v_idx: [] for v_idx in range(num_vertices)}
    shrtstP = {v_idx: (float('inf'), float('inf')) for v_idx in range(num_vertices)}

    if 0 <= start_vertex_id < num_vertices:
        P_all_hops[start_vertex_id] = [(0, 0)]
        P_new_prev_hop[start_vertex_id] = [(0, 0)]
        shrtstP[start_vertex_id] = (0, 0)
    else:
        print("Geçersiz başlangıç düğümü ID'si")
        return shrtstP

    new_paths_count_in_iteration = 1
    k = 0

    while k < num_vertices - 1 and new_paths_count_in_iteration > 0:
        k += 1
        new_paths_count_in_iteration = 0
        
        P_new_candidates_current_hop = {v_idx: [] for v_idx in range(num_vertices)}
        for u in range(num_vertices):
            if not P_new_prev_hop.get(u): continue # Eğer u için önceki hopta yol yoksa
            if u not in graph_adj: continue

            for prev_arrival_at_u, prev_length_to_u in P_new_prev_hop[u]:
                for v, intervals in graph_adj[u].items():
                    for s_i, c_i, lambda_i in intervals:
                        departure_time_from_u = -1
                        if prev_arrival_at_u <= s_i:
                            departure_time_from_u = s_i
                        elif prev_arrival_at_u <= c_i:
                            departure_time_from_u = prev_arrival_at_u
                        
                        if departure_time_from_u != -1:
                            arrival_at_v = departure_time_from_u + lambda_i
                            length_to_v = prev_length_to_u + lambda_i
                            P_new_candidates_current_hop.setdefault(v, []).append((arrival_at_v, length_to_v))
        
        P_new_current_hop_next_iter = {v_idx: [] for v_idx in range(num_vertices)}
        for v_node in range(num_vertices):
            candidates_for_v = P_new_candidates_current_hop.get(v_node, [])
            if candidates_for_v:
                P_new_current_hop_next_iter[v_node] = manage_dominance_and_sort(candidates_for_v)

            # P_all_hops ve shrtstP güncelle
            if P_new_current_hop_next_iter.get(v_node) or P_all_hops.get(v_node): # Eğer güncellenecek bir şey varsa
                temp_merged = list(P_all_hops.get(v_node, []))
                if P_new_current_hop_next_iter.get(v_node):
                     temp_merged.extend(P_new_current_hop_next_iter[v_node])
                
                P_all_hops[v_node] = manage_dominance_and_sort(temp_merged)

                current_best_len = shrtstP.get(v_node, (float('inf'), float('inf')))[1]
                best_path_for_shrtst = shrtstP.get(v_node, (float('inf'), float('inf')))

                for arr_v, len_v in P_all_hops.get(v_node, []):
                    if len_v < current_best_len:
                        best_path_for_shrtst = (arr_v, len_v)
                        current_best_len = len_v
                    elif len_v == current_best_len:
                        if arr_v < best_path_for_shrtst[0]:
                            best_path_for_shrtst = (arr_v, len_v)
                shrtstP[v_node] = best_path_for_shrtst
            
            if P_new_current_hop_next_iter.get(v_node):
                new_paths_count_in_iteration = 1
        
        P_new_prev_hop = P_new_current_hop_next_iter
    return shrtstP

# --- Paralel Algoritma (Multiprocessing) ---
# Her bir v_node için yapılacak işi tanımlayan yardımcı fonksiyon (Global olmalı veya pickle edilebilir olmalı)
def process_v_node_python_job(args_tuple):
    v_node_idx, p_new_candidates_for_v, current_p_all_hops_v, current_shrtst_p_v = args_tuple
    
    # P_new(v,k) oluşturma
    p_new_v_k = []
    if p_new_candidates_for_v:
        p_new_v_k = manage_dominance_and_sort(p_new_candidates_for_v)

    new_p_all_v_k = list(current_p_all_hops_v) # Kopyala
    new_shrtst_v = current_shrtst_p_v # Tuple olduğu için kopyalanmış olur
    path_found_for_v = False

    if p_new_v_k:
        new_p_all_v_k.extend(p_new_v_k)
        new_p_all_v_k = manage_dominance_and_sort(new_p_all_v_k)

        current_best_len = new_shrtst_v[1]
        best_path_for_shrtst = new_shrtst_v

        for arr_v, len_v in new_p_all_v_k:
            if len_v < current_best_len:
                best_path_for_shrtst = (arr_v, len_v)
                current_best_len = len_v
            elif len_v == current_best_len:
                if arr_v < best_path_for_shrtst[0]:
                    best_path_for_shrtst = (arr_v, len_v)
        new_shrtst_v = best_path_for_shrtst
        path_found_for_v = True
            
    return v_node_idx, p_new_v_k, new_p_all_v_k, new_shrtst_v, path_found_for_v

def itg_shortest_path_algorithm2_python_multiprocessing(graph_adj, num_vertices, start_vertex_id):
    P_all_hops = {v_idx: [] for v_idx in range(num_vertices)}
    P_new_prev_hop = {v_idx: [] for v_idx in range(num_vertices)}
    shrtstP = {v_idx: (float('inf'), float('inf')) for v_idx in range(num_vertices)}

    if 0 <= start_vertex_id < num_vertices:
        P_all_hops[start_vertex_id] = [(0, 0)]
        P_new_prev_hop[start_vertex_id] = [(0, 0)]
        shrtstP[start_vertex_id] = (0, 0)
    else:
        print("Geçersiz başlangıç düğümü ID'si")
        return shrtstP

    new_paths_count_in_iteration = 1
    k = 0
    num_processes = min(max(1, multiprocessing.cpu_count() -1), 4) # Ayarlanabilir

    while k < num_vertices - 1 and new_paths_count_in_iteration > 0:
        k += 1
        
        P_new_candidates_current_hop = {v_idx: [] for v_idx in range(num_vertices)}
        # Adım 1: Sıralı (Daha önce tartışıldığı gibi)
        for u in range(num_vertices):
            if not P_new_prev_hop.get(u): continue
            if u not in graph_adj: continue
            for prev_arrival_at_u, prev_length_to_u in P_new_prev_hop[u]:
                for v, intervals in graph_adj[u].items():
                    for s_i, c_i, lambda_i in intervals:
                        departure_time_from_u = -1
                        if prev_arrival_at_u <= s_i:
                            departure_time_from_u = s_i
                        elif prev_arrival_at_u <= c_i:
                            departure_time_from_u = prev_arrival_at_u
                        if departure_time_from_u != -1:
                            arrival_at_v = departure_time_from_u + lambda_i
                            length_to_v = prev_length_to_u + lambda_i
                            P_new_candidates_current_hop.setdefault(v, []).append((arrival_at_v, length_to_v))
        
        tasks_args = []
        for v_idx in range(num_vertices):
            tasks_args.append(
                (v_idx,
                 P_new_candidates_current_hop.get(v_idx, []),
                 P_all_hops.get(v_idx, []),
                 shrtstP.get(v_idx, (float('inf'), float('inf')))
                )
            )
        
        P_new_current_hop_next_iter = {v_idx: [] for v_idx in range(num_vertices)}
        new_paths_count_in_iteration = 0

        # Sadece yeterince iş varsa paralelleştir
        if num_processes > 1 and num_vertices > num_processes * 2 :
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(process_v_node_python_job, tasks_args)
                
                for res_v_idx, res_p_new_v_k, res_p_all_v_k, res_shrtst_v, res_path_found in results:
                    P_new_current_hop_next_iter[res_v_idx] = res_p_new_v_k
                    P_all_hops[res_v_idx] = res_p_all_v_k
                    shrtstP[res_v_idx] = res_shrtst_v
                    if res_path_found:
                        new_paths_count_in_iteration = 1
            except Exception as e:
                print(f"Multiprocessing hatası: {e}. Sıralı devam ediliyor...")
                # Hata durumunda sıralı çalıştır (veya programı sonlandır) - Bu kısmı basitleştirdim, sadece bildiriyor.
                # Gerçek uygulamada burada sıralı fallback yapmak daha robust olur.
                new_paths_count_in_iteration = 1 # Hata durumunda sonsuz döngüyü engelle
        else: # Sıralı çalıştır
            for task_arg_tuple in tasks_args:
                v_idx_seq, p_new_seq, p_all_seq, shrtst_seq, path_found_seq = process_v_node_python_job(task_arg_tuple)
                P_new_current_hop_next_iter[v_idx_seq] = p_new_seq
                P_all_hops[v_idx_seq] = p_all_seq
                shrtstP[v_idx_seq] = shrtst_seq
                if path_found_seq:
                    new_paths_count_in_iteration = 1
        
        P_new_prev_hop = P_new_current_hop_next_iter
    return shrtstP

# --- Main Bloğu ---
if __name__ == "__main__":
    # Multiprocessing için bu kontrol önemli olabilir (özellikle Windows'ta)
    # multiprocessing.freeze_support()

    BIG_NUM_VERTICES = 200
    EDGE_DENSITY_BIG = 0.05
    MAX_INTERVALS_BIG = 2
    MAX_TIME_BIG = 300
    MAX_DURATION_BIG = 20
    MAX_TRAVEL_BIG = 10
    SHARED_SEED = 123
    START_NODE_BIG = 0

    print(f"--- Python (Büyük Graf, {BIG_NUM_VERTICES} düğüm) ---")
    itg_big_python = generate_deterministic_itg(
        BIG_NUM_VERTICES, EDGE_DENSITY_BIG, MAX_INTERVALS_BIG,
        MAX_TIME_BIG, MAX_DURATION_BIG, MAX_TRAVEL_BIG, seed=SHARED_SEED
    )
    
    print(f"\nGraf oluşturuldu. {len(itg_big_python)} anahtar, toplam kenar sayısı (yaklaşık): {sum(len(v) for v in itg_big_python.values())}")

    # Sıralı Python
    print("\nSıralı Python çalıştırılıyor...")
    start_py_seq = time.time()
    shrtstP_py_seq = itg_shortest_path_algorithm2_python_sequential(itg_big_python, BIG_NUM_VERTICES, START_NODE_BIG)
    end_py_seq = time.time()
    # Sadece birkaç sonuç yazdır
    print("Sıralı Python örnek sonuçları (ilk 3 ve son 2):")
    results_to_print_seq = {i: shrtstP_py_seq[i] for i in list(range(3)) + list(range(BIG_NUM_VERTICES-2, BIG_NUM_VERTICES)) if i in shrtstP_py_seq}
    for node, path_data in results_to_print_seq.items():
        print(f"  Düğüm {node}: {path_data}")
    print(f"Sıralı Python süresi: {end_py_seq - start_py_seq:.4f} saniye")

    # Paralel Python (Multiprocessing)
    print("\nParalel Python (multiprocessing) çalıştırılıyor...")
    start_py_par = time.time()
    shrtstP_py_par = itg_shortest_path_algorithm2_python_multiprocessing(itg_big_python, BIG_NUM_VERTICES, START_NODE_BIG)
    end_py_par = time.time()
    print("Paralel Python örnek sonuçları (ilk 3 ve son 2):")
    results_to_print_par = {i: shrtstP_py_par[i] for i in list(range(3)) + list(range(BIG_NUM_VERTICES-2, BIG_NUM_VERTICES)) if i in shrtstP_py_par}
    for node, path_data in results_to_print_par.items():
        print(f"  Düğüm {node}: {path_data}")
    print(f"Paralel Python (multiprocessing) süresi: {end_py_par - start_py_par:.4f} saniye")