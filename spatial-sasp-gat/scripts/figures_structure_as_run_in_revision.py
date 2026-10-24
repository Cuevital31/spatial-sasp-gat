# Figure generation cell exactly as run in the revision (Fig. 3-5 / S6 representative slides). NOT re-run or verified by the notebooks:
# it needs umap-learn, anndata and the locked protocol/manifest files in Drive; figures are qualitative. Kept for transparency.
# HÜCRE 9-C REV v3 — Paket 9: şekiller (kilit: package9_protocol_v1.json; profiller: recon_profiles_v1; metrikler: 9-B CSV'leri).
# Temsilci slide'lar kilitte sabit: MISC40 (TEST), TENX180 (HD); Full seed 42, k=5. Eğitim YOK, yeni metrik YOK, slide/ayar seçimi YOK.
# Üretilenler: (1) slide başına küme haritaları + ölçülmüş-gen SASP skor haritaları + UMAP (gözlenen PCA üzerinde fit; diğerleri transform), (2) slide başına ısı haritası, (3) kohort düzeyi eşleştirilmiş Full - Ring2Mean nokta grafiği.
# Hungarian eşleştirme yalnızca renk hizalar; gösterilen ARI değerleri 9-B CSV'siyle karşılaştırılarak doğrulanır (yeniden hesaplanan etiketler tutarlılık denetimi içindir).
# REV v3: h5ad klasörü hest_expression/st, dosya seçimi tam ad ({slide}.h5ad); REV v2 düzeltmeleri korunur (lo<hi, doygunluk açıklaması, ısı haritası sunumu, grup bazlı y eksenleri).
# Yayın: bellekte üret -> geçici dosya -> geri okuma -> dışlayıcı oluşturma -> sha256; manifest en sonda. Drive atomik yeniden adlandırma garantisi vermez.
import io, os, re, json, hashlib, platform, time, shutil, warnings, importlib.metadata as imd
from pathlib import Path
import numpy as np, pandas as pd

def figures_main():
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    from IPython.display import display, Image
    try: import umap
    except Exception as e: raise RuntimeError("umap-learn kurulu değil. Ayrı bir hücrede '!pip install umap-learn' çalıştırıp bu hücreyi yeniden çalıştırın (bu hücre paket kurmaz).") from e
    try: import anndata
    except Exception as e: raise RuntimeError("anndata kurulu değil. Ayrı bir hücrede '!pip install anndata' çalıştırıp bu hücreyi yeniden çalıştırın.") from e

    BASE = Path('/content/drive/MyDrive/New-SASP-idea'); OUT9 = BASE / 'revision_reconstruction_2026-10-20'; assert OUT9.is_dir(), 'Drive bağlı değil veya klasör yok'
    LOCK = OUT9 / 'package9_protocol_v1.json'; MAN = OUT9 / 'recon_profiles_manifest_v1.json'; PDIR = OUT9 / 'recon_profiles_v1'
    F_SLIDE = OUT9 / 'reconstruction_slide_metrics_v1.csv'; F_COH = OUT9 / 'reconstruction_cohort_summary_v1.csv'; F_MMAN = OUT9 / 'reconstruction_metrics_manifest_v1.json'
    FDIR = OUT9 / 'figures_v1'; QDIR = OUT9 / 'figures_quarantine'; F_FMAN = FDIR / 'figures_manifest_v1.json'; HDIR = BASE / 'hest_expression' / 'st'; assert HDIR.is_dir(), f'h5ad klasörü yok: {HDIR}'
    LOCK_FILE_SHA = '5130c0e9f728458eb0749023ea4bc22decf2114a03e98386b6dbc340e15757a3'; LOCK_BODY_SHA = '02ebd504fe0a7d5bc142b6bd783a19d64b2eb696a955c42a999fae7da46597ae'
    MAN_SHA = '24b342fdd3cd965b5cba8a4e3b5348812ab735e84b3d3870585d356468cfa675'; MMAN_SHA = '53a55a2f32f2a4c509f90728d1f2a8839a442acd8578c1f20279e40f67b5e8c4'
    REP = [('test', 'MISC40'), ('hd', 'TENX180')]; SEED = 42; K = 5; FP = f'full_s{SEED}'; PAL = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#6A3D9A']
    COH_LAB = {'test': 'TEST', 'hd': 'External HD'}
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
        return h.hexdigest()
    shab = lambda b: hashlib.sha256(b).hexdigest()

    # ---------- tamamlanmış çalışma ----------
    if F_FMAN.exists():
        fm = json.load(open(F_FMAN, encoding='utf-8'))
        assert fm['protocol_body_sha256'] == LOCK_BODY_SHA and fm['metrics_manifest_sha256'] == MMAN_SHA, 'şekil manifesti girdilerle uyuşmuyor'
        bad = [n for n, h in fm['outputs'].items() if not (FDIR / n).is_file() or sha(FDIR / n) != h]; assert not bad, f'şekil dosyaları manifestle uyuşmuyor: {bad}'
        print(f'9-C zaten tamamlanmış: {len(fm["outputs"])} dosyanın sha256 değeri manifestle aynı. Manifest sha256: {sha(F_FMAN)}'); return

    print('A) Girdi doğrulaması')
    assert sha(LOCK) == LOCK_FILE_SHA and sha(MAN) == MAN_SHA and sha(F_MMAN) == MMAN_SHA, 'kilit, 9-A manifesti veya 9-B manifesti değişmiş'
    lock = json.load(open(LOCK, encoding='utf-8')); man = json.load(open(MAN, encoding='utf-8')); mman = json.load(open(F_MMAN, encoding='utf-8'))
    assert lock['protocol_body_sha256'] == LOCK_BODY_SHA and mman['protocol_body_sha256'] == LOCK_BODY_SHA
    for n, h in mman['outputs'].items(): assert (OUT9 / n).is_file() and sha(OUT9 / n) == h, f'9-B çıktısı manifestle uyuşmuyor: {n}'
    figtxt = json.dumps(lock['figures']); assert all(s_ in figtxt for _, s_ in REP), 'kilitteki şekil bölümü MISC40/TENX180 temsilci slide\'larını içermiyor'
    SL = [(c, x['slide']) for c in ('test', 'hd') for x in lock['scope']['slides'][c]]; assert all(r in SL for r in REP)
    mu_tr = np.array(lock['feature_space']['train_gene_mean'], dtype=np.float64); sd_tr = np.array(lock['feature_space']['train_gene_sd'], dtype=np.float64); gene_order = lock['feature_space']['gene_order']
    assert mu_tr.shape == (54,) and (sd_tr > 0).all() and len(gene_order) == 54
    D = pd.read_csv(F_SLIDE, float_precision='round_trip'); CS = pd.read_csv(F_COH, float_precision='round_trip'); assert len(D) == 1000 and len(CS) == 44
    D['kk'] = D.k.fillna(-1).astype(int); CS['kk'] = CS.k.fillna(-1).astype(int)
    print('   kilit, 9-A ve 9-B manifestleri ve çıktıları doğrulandı; temsilci slide\'lar kilitle uyumlu')

    def load_prof(sid, p):
        name = f'{sid}_{p}.npz'; assert sha(PDIR / name) == man['files_sha256'][name], f'{name}: sha256 uyuşmuyor'
        z = np.load(PDIR / name, allow_pickle=False)
        with z:
            assert str(z['protocol_body_sha256']) == LOCK_BODY_SHA and str(z['slide_id']) == sid and str(z['profile']) == p
            gi = z['gene_indices'].astype(int); on = z['obs_names'].astype(str); X = (z['observed_values'] if p == 'observed' else z['reconstructed_values']).astype(np.float64)
        return gi, on, X
    def get_coords(sid, obs_names):
        path = HDIR / f'{sid}.h5ad'; assert path.is_file(), f'{sid}: {path} bulunamadı'
        ad = anndata.read_h5ad(path, backed='r')
        try:
            h5_names = np.array(ad.obs_names).astype(str); xy = np.array(ad.obsm['spatial'], dtype=np.float64)
        finally:
            try: ad.file.close()
            except Exception: pass
        assert len(set(h5_names)) == len(h5_names) and xy.ndim == 2 and xy.shape == (len(h5_names), 2) and np.isfinite(xy).all(), f'{sid}: koordinat/obs_names yapısı geçersiz'
        idx = pd.Index(h5_names).get_indexer(obs_names); assert (idx >= 0).all(), f'{sid}: PKL spotlarının tamamı h5ad içinde değil'
        assert len(h5_names) == len(obs_names) and np.array_equal(idx, np.arange(len(obs_names))), f'{sid}: obs_names sırası PKL ile birebir aynı değil; şekil üretilmeyecek'
        return xy[idx], path.name
    def match_colors(ref, lab, k):
        assert sorted(np.unique(ref)) == list(range(k)) and sorted(np.unique(lab)) == list(range(k)), 'etiket kümesi eksik'
        M = np.zeros((k, k), int); np.add.at(M, (ref, lab), 1); r, c = linear_sum_assignment(-M); mp = {int(cc): int(rr) for rr, cc in zip(r, c)}; assert len(mp) == k
        return np.array([mp[int(l)] for l in lab])
    def spot_size(xy, panel_in=3.3):
        d, _ = cKDTree(xy).query(xy, k=2); return (0.9 * np.median(d[:, 1]) / np.ptp(xy, axis=0).max() * panel_in * 72) ** 2
    def render(fig, name, png_dpi=200):
        out = {}
        b = io.BytesIO(); fig.savefig(b, format='png', dpi=png_dpi, metadata={'Software': None}); out[f'{name}.png'] = b.getvalue()
        b = io.BytesIO(); fig.savefig(b, format='pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None}); out[f'{name}.pdf'] = b.getvalue()
        return out

    FILES = {}; META = {}; CHECK = []
    print('B) Temsilci slide\'lar: profiller, PCA, k-means, UMAP, şekiller')
    for coh, sid in REP:
        P = {}; ref = None
        for p in ('observed', FP, 'ring2mean'):
            gi, on, X = load_prof(sid, p)
            if ref is None: ref = (gi, on)
            assert np.array_equal(gi, ref[0]) and np.array_equal(on, ref[1]), f'{sid} {p}: gene_indices/obs_names profiller arasında farklı'
            P[p] = (X - mu_tr[gi]) / sd_tr[gi]; assert np.isfinite(P[p]).all()
        gi, on = ref; n, ng = P['observed'].shape; genes = [gene_order[j] for j in gi]
        xy, h5name = get_coords(sid, on)
        pca = PCA(n_components=10, svd_solver='full', random_state=0).fit(P['observed']); T = {p: pca.transform(P[p]) for p in P}
        lab = {p: KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(T[p]) for p in P}
        ari = {}
        for p in (FP, 'ring2mean'):
            r_ = D[(D.slide == sid) & (D.profile == p) & (D.metric == 'ARI') & (D.kk == K)]; assert len(r_) == 1
            a_new = float(adjusted_rand_score(lab['observed'], lab[p])); a_csv = float(r_.value.iloc[0]); assert abs(a_new - a_csv) < 1e-9, f'{sid} {p}: ARI 9-B CSV\'siyle uyuşmuyor ({a_new} / {a_csv}); şekil üretilmeyecek'
            ari[p] = a_csv; CHECK.append(dict(slide=sid, profile=p, ari_recomputed=round(a_new, 6), ari_9B_csv=round(a_csv, 6)))
        mlab = {'observed': lab['observed'], FP: match_colors(lab['observed'], lab[FP], K), 'ring2mean': match_colors(lab['observed'], lab['ring2mean'], K)}   # yalnızca görüntü
        S = {p: P[p].mean(axis=1) for p in P}; lo, hi = np.percentile(S['observed'], [1, 99]); assert np.isfinite(lo) and np.isfinite(hi) and lo < hi, f'{sid}: skor renk skalası geçersiz (lo={lo}, hi={hi})'   # ortak ölçek yalnızca gözlenen profilden
        sat = {p: float(((S[p] < lo) | (S[p] > hi)).mean()) for p in S}                                                                                                                     # tanısal: skala dışına taşan spot oranı (sunum bilgisi)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0); emb = {'observed': reducer.fit_transform(T['observed'])}
            for p in (FP, 'ring2mean'): emb[p] = reducer.transform(T[p])
        allE = np.vstack(list(emb.values())); pad = 0.05 * np.ptp(allE, axis=0); xl = (allE[:, 0].min() - pad[0], allE[:, 0].max() + pad[0]); yl = (allE[:, 1].min() - pad[1], allE[:, 1].max() + pad[1])
        s_ = spot_size(xy); pc = np.array(PAL)
        names = [('observed', 'Observed'), (FP, f'Full SASP-GAT (seed {SEED})'), ('ring2mean', 'Ring2Mean')]
        # ---- Şekil 1: küme haritası / skor haritası / UMAP ----
        fig = Figure(figsize=(10.6, 10.8), layout='constrained'); axs = fig.subplots(3, 3); mp_ = None
        for j, (p, nm) in enumerate(names):
            ax = axs[0, j]; ax.scatter(xy[:, 0], xy[:, 1], c=pc[mlab[p]], s=s_, linewidths=0, rasterized=True); ttl = f'{nm}\nk-means, k={K}' if p == 'observed' else f'{nm}\nARI vs observed = {ari[p]:.3f}'
            ax.set_title(ttl, fontsize=10)
            ax = axs[1, j]; mp_ = ax.scatter(xy[:, 0], xy[:, 1], c=S[p], s=s_, linewidths=0, cmap='viridis', vmin=lo, vmax=hi, rasterized=True); ax.set_title(nm, fontsize=10)
            for r_ in (0, 1): axs[r_, j].set_aspect('equal'); axs[r_, j].invert_yaxis(); axs[r_, j].axis('off')
            ax = axs[2, j]; ax.scatter(emb[p][:, 0], emb[p][:, 1], c=pc[mlab[p]], s=4, linewidths=0, alpha=0.7, rasterized=True); ax.set_xlim(xl); ax.set_ylim(yl); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(nm if p == 'observed' else f'{nm}\n(projected onto observed UMAP)', fontsize=10); ax.set_xlabel('UMAP 1', fontsize=8)
        axs[2, 0].set_ylabel('UMAP 2', fontsize=8)
        for r_, t_ in enumerate(['Spatial clusters', 'Mean standardized SASP-expression score', 'UMAP (fit on observed PCA)']): axs[r_, 0].text(-0.06, 0.5, t_, transform=axs[r_, 0].transAxes, rotation=90, va='center', ha='right', fontsize=10)
        fig.colorbar(mp_, ax=axs[1, :], shrink=0.7, label='mean TRAIN z-score over measured SASP genes\n(colour scale: observed 1st–99th percentile; values outside are saturated)')
        fig.legend([Line2D([], [], marker='o', ls='', color=c_) for c_ in PAL], [f'cluster {i + 1}' for i in range(K)], loc='outside lower center', ncol=K, fontsize=9, title='Reconstructed-cluster colours matched to observed clusters (Hungarian assignment; display only)', title_fontsize=8)
        fig.suptitle(f'{sid} ({COH_LAB[coh]}) — {n} spots, {ng} measured SASP genes', fontsize=12)
        FILES.update(render(fig, f'fig9_{sid}_k{K}_seed{SEED}_maps_umap')); display(Image(data=FILES[f'fig9_{sid}_k{K}_seed{SEED}_maps_umap.png'], width=900))
        # ---- Şekil 2: ısı haritası (gözlenen k=5 etiketi, sonra gözlenen skor ile sıralı; kilitteki gen sırası; tam çözünürlük) ----
        order = np.lexsort((S['observed'], lab['observed'])); fig = Figure(figsize=(12, 19), layout='constrained'); gs = fig.add_gridspec(4, 2, height_ratios=[0.25, 5, 5, 5], width_ratios=[40, 1])
        st = fig.add_subplot(gs[0, 0]); st.imshow(lab['observed'][order][None, :], aspect='auto', cmap=ListedColormap(PAL), vmin=-0.5, vmax=K - 0.5, interpolation='nearest'); st.set_xticks([]); st.set_yticks([]); st.set_title('Observed k-means clusters (k=5); the same spot order is used in all panels', fontsize=10)
        im = None
        for r_, (p, nm) in enumerate(names):
            ax = fig.add_subplot(gs[r_ + 1, 0]); im = ax.imshow(np.clip(P[p][order].T, -3, 3), aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='antialiased', rasterized=True)
            ax.set_yticks(range(ng)); ax.set_yticklabels(genes, fontsize=6); ax.set_xticks([]); ax.set_title(nm, fontsize=11)
            if r_ == 2: ax.set_xlabel(f'{n} spots (ordered by observed cluster, then observed score)', fontsize=9)
        fig.colorbar(im, cax=fig.add_subplot(gs[1:, 1]), label='TRAIN z-score (clipped to [-3, 3])')
        fig.suptitle(f'{sid} ({COH_LAB[coh]}) — {ng} measured SASP genes × {n} spots', fontsize=12)
        FILES.update(render(fig, f'fig9_{sid}_k{K}_seed{SEED}_heatmap', png_dpi=300)); display(Image(data=FILES[f'fig9_{sid}_k{K}_seed{SEED}_heatmap.png'], width=700))
        META[sid] = dict(cohort=coh, n_spots=int(n), n_measured_genes=int(ng), h5ad=h5name, ari_vs_observed={p: ari[p] for p in ari}, seed=SEED, k=K, score_colour_scale=dict(source='observed 1st-99th percentile', lo=float(lo), hi=float(hi), fraction_outside_scale=sat),
                         umap=dict(n_neighbors=15, min_dist=0.1, random_state=0, fit_on='observed_PCA10', others='transform'), heatmap=dict(figsize_in=[12, 19], png_dpi=300, all_spots_all_genes=True))
        print(f'   {sid}: skor skalası [{lo:.3f}, {hi:.3f}]; skala dışına taşan spot oranı — ' + ', '.join(f'{p}: {sat[p]:.3f}' for p in S))

    # ---------- Şekil 3: kohort düzeyi eşleştirilmiş fark grafiği (yalnızca 9-B CSV'leri) ----------
    print('C) Kohort düzeyi eşleştirilmiş fark grafiği (9-B CSV\'lerinden)')
    METS = [('ARI', 3, 'ARI, k=3'), ('ARI', 5, 'ARI, k=5'), ('ARI', 7, 'ARI, k=7'), ('PCA_R2', None, 'PCA-space R²'), ('score_pearson', None, 'Score Pearson'), ('score_spearman', None, 'Score Spearman')]
    DD = D[D.profile == 'full_minus_ring2']; CC = CS[CS.profile == 'full_minus_ring2']; assert len(DD) == 150 and len(CC) == 12
    fig = Figure(figsize=(14, 4.8), layout='constrained'); axs = fig.subplots(1, 6); axs[1].sharey(axs[0]); axs[2].sharey(axs[0]); axs[5].sharey(axs[4])   # ARI üçlüsü ve skor korelasyonu ikilisi ortak eksen; PCA R2 ayrı
    rng = np.random.default_rng(0); NSL = {'test': 13, 'hd': 12}
    for ax, (m, k, lb) in zip(axs, METS):
        kk = -1 if k is None else k
        for xi, coh in enumerate(('test', 'hd')):
            sel = DD[(DD.cohort == coh) & (DD.metric == m) & (DD.kk == kk)]; assert len(sel) == NSL[coh] and (sel.status == 'ok').all()
            cr = CC[(CC.cohort == coh) & (CC.metric == m) & (CC.kk == kk)]; assert len(cr) == 1; cr = cr.iloc[0]
            v = sel.value.to_numpy(float); jit = rng.uniform(-0.12, 0.12, len(v))
            assert abs(np.median(v) - cr['median']) < 1e-9 and int((v > 0).sum()) == int(cr.n_full_better) and int((v < 0).sum()) == int(cr.n_ring2_better) and int((v == 0).sum()) == int(cr.n_equal) and len(v) == int(cr.n_comparable), 'kohort özeti ile slide düzeyi değerler tutarsız'
            for msk, mk, col in ((v > 0, 'o', '#0072B2'), (v < 0, 'D', '#D55E00'), (v == 0, 's', '#777777')):
                if msk.any(): ax.scatter(xi + jit[msk], v[msk], marker=mk, c=col, s=26, linewidths=0, alpha=0.9, zorder=3)
            ax.hlines(cr['median'], xi - 0.25, xi + 0.25, color='k', lw=2, zorder=4); ax.vlines(xi + 0.32, cr.q25, cr.q75, color='#555555', lw=3, zorder=2)
            ax.text(xi, 0.015, f'{int(cr.n_full_better)}/{int(cr.n_comparable)}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8)
        ax.axhline(0, ls='--', color='grey', lw=1, zorder=1); ax.set_xticks([0, 1]); ax.set_xticklabels([f'TEST\n(n={NSL["test"]})', f'External HD\n(n={NSL["hd"]})'], fontsize=8); ax.set_xlim(-0.6, 1.7); ax.set_title(lb, fontsize=10)
    axs[0].set_ylabel('Full − Ring2Mean (paired, per slide)'); axs[3].set_ylabel('Full − Ring2Mean (paired, per slide)', fontsize=8); axs[4].set_ylabel('Full − Ring2Mean (paired, per slide)', fontsize=8)
    fig.legend([Line2D([], [], marker='o', ls='', color='#0072B2'), Line2D([], [], marker='D', ls='', color='#D55E00'), Line2D([], [], color='k', lw=2), Line2D([], [], color='#555555', lw=3)],
               ['Full better', 'Ring2Mean better', 'median', 'IQR'], loc='outside lower center', ncol=4, fontsize=9,
               title='Numbers: slides where Full is better / comparable slides (Full = mean of 3 seeds); each point is one slide; y axes are shared within the ARI panels and within the correlation panels only', title_fontsize=8)
    fig.suptitle('Paired slide-level differences: Full SASP-GAT minus Ring2Mean (higher = Full better)', fontsize=11)
    FILES.update(render(fig, 'fig9_cohort_paired_differences')); display(Image(data=FILES['fig9_cohort_paired_differences.png'], width=1000))

    # ---------- yayın ----------
    FDIR.mkdir(exist_ok=True); QUAR = []
    def quarantine(p, why):
        QDIR.mkdir(exist_ok=True); dst = QDIR / f'{p.name}.{time.strftime("%Y%m%d_%H%M%S")}'; k_ = 0
        while dst.exists(): k_ += 1; dst = QDIR / f'{p.name}.{time.strftime("%Y%m%d_%H%M%S")}_{k_}'
        old = sha(p); shutil.move(str(p), str(dst)); assert dst.exists() and sha(dst) == old, f'Karantina taşıma bütünlüğü doğrulanamadı: {p.name}'
        QUAR.append(dict(file=p.name, moved_to=dst.name, sha256=old, reason=why)); print(f'   !! KARANTİNA: {p.name} -> {QDIR.name}/{dst.name} ({why}; sha256 doğrulandı)')
    for p in sorted(FDIR.glob('.tmp_*')): quarantine(p, 'önceki çalıştırmadan kalan geçici dosya')
    def publish(final, data):
        if final.exists():
            if sha(final) == shab(data): print(f'   {final.name}: mevcut dosya aynı, korundu'); return
            quarantine(final, 'bellekteki içerikten farklı (yarım kalmış veya eski çıktı)')
        tmp = final.with_name('.tmp_' + final.name)
        try:
            with open(tmp, 'xb') as f: f.write(data); f.flush()
            assert tmp.read_bytes() == data, 'geçici dosya geri okuma uyuşmuyor'
            with open(final, 'xb') as f:
                f.write(data); f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
            assert sha(final) == shab(data), 'nihai dosya sha256 bellekteki içerikle aynı değil'
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
        print(f'   Yayımlandı: {final.name} ({len(data) / 1024:.0f} KB, sha256 {shab(data)[:12]}…)')
    for nm_ in sorted(FILES): publish(FDIR / nm_, FILES[nm_])
    def v(n):
        try: return imd.version(n)
        except Exception: return None
    fm_ = dict(created='2026-10-20', protocol_file_sha256=LOCK_FILE_SHA, protocol_body_sha256=LOCK_BODY_SHA, profiles_manifest_sha256=MAN_SHA, metrics_manifest_sha256=MMAN_SHA,
               outputs={n_: shab(b_) for n_, b_ in sorted(FILES.items())}, representative_slides=META, ari_consistency_check=CHECK, quarantined=QUAR,
               libraries={n_: v(n_) for n_ in ['matplotlib', 'umap-learn', 'scikit-learn', 'scipy', 'numpy', 'pandas', 'anndata']}, python=platform.python_version(),
               atomicity_note='Google Drive gives no atomic-rename guarantee; files were published by temp-file write, read-back, exclusive create and sha256 equality; this manifest is written last',
               note='Representative slides, seed and k fixed in the locked protocol; no result-dependent selection. UMAP is qualitative; colours of reconstructed clusters are Hungarian-matched to observed clusters for display only; score maps share the observed 1st-99th percentile colour scale (values outside are saturated); no new metrics.')
    publish(F_FMAN, json.dumps(fm_, indent=2, sort_keys=True).encode('utf-8'))
    print(f'\nD) ARI tutarlılık denetimi (şekilde gösterilen değer = 9-B CSV; yeniden üretim farkı < 1e-9)'); print(pd.DataFrame(CHECK).to_string(index=False))
    print(f'\n{len(FILES)} şekil dosyası + manifest yazıldı | karantina: {len(QUAR)} | manifest sha256: {sha(F_FMAN)}')
    print('UMAP yalnızca niteldir; küme renkleri gösterim amaçlı eşleştirilmiştir; ölçülmüş-gen skoru doğrulanmış bir senesens skoru değildir.')
figures_main()