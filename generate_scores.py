
import os
import glob
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

# Setup: Seed for reproducibility
np.random.seed(42)

def main():
    """
    Main function to run the entire scoring pipeline.
    """
    # Define paths
    dataset_root = Path("/HDD_16T/rsy/UEDG-master/dataset/")
    output_dir = Path("./Ranking_whole/")
    output_dir.mkdir(exist_ok=True)

    # Output file paths
    csv_path = output_dir / "gt_continuous.csv"
    weights_path = output_dir / "bradley_terry_weights.pkl"
    ecdf_params_path = output_dir / "ecdf_params.json"
    diagnostics_png_path = output_dir / "score_diagnostics.png"
    diagnostics_pdf_path = output_dir / "score_diagnostics.pdf"
    spearman_metrics_path = output_dir / "spearman_metrics.json"

    # Run pipeline
    print("--- Starting Data Processing Pipeline (Scheme B-soft) ---")

    # Step 0: Enumerate files and get metadata
    all_files = get_file_list(dataset_root)
    if not all_files:
        print("No image/mask files found. Please check dataset paths.")
        return
    
    data_df = pd.DataFrame(all_files)
    # Use absolute path as a unique ID for each mask
    data_df['mask_id'] = data_df['mask_path'].apply(lambda p: str(p.resolve()))
    print(f"Found {len(data_df)} total valid mask files.")

    # Step 1: Build comparison list for Bradley–Terry (BT)
    print("\n--- Step 1: Building Bradley-Terry comparison list ---")
    comparisons = build_comparisons(data_df, seed=42)
    print(f"Generated {len(comparisons)} comparison pairs.")

    # Step 2: Fit Bradley–Terry model
    print("\n--- Step 2: Fitting Bradley-Terry model ---")
    weights = fit_bradley_terry(comparisons, pd.Series(data_df['mask_id'].unique()).to_numpy())
    if weights is None:
        print("Bradley-Terry model fitting failed. Exiting.")
        return
    
    weights_df = pd.DataFrame(list(weights.items()), columns=['mask_id', 'w'])
    data_df = pd.merge(data_df, weights_df, on='mask_id')
    
    # Save raw weights
    with open(weights_path, "wb") as f:
        joblib.dump(weights, f)
    print(f"Saved Bradley-Terry weights to {weights_path}")

    # Step 3: Center & map to (0,1)
    print("\n--- Step 3: Post-processing weights ---")
    data_df, ecdf_params = post_process_weights(data_df, gamma=0.8, seed=42)
    
    # Save ECDF parameters
    with open(ecdf_params_path, "w") as f:
        json.dump(ecdf_params, f, indent=4)
    print(f"Saved ECDF parameters to {ecdf_params_path}")

    # Step 4: Final score
    print("\n--- Step 4: Mapping to final scores ---")
    data_df['score'] = data_df['z'] * 10
    
    # Step 6: Diagnostics & Stop Criteria
    print("\n--- Step 6: Calculating diagnostics ---")
    metrics, warnings_log = calculate_diagnostics(data_df)
    
    print(f"Spearman ρ (COD) : {metrics['rho_cod']:.4f}")
    print(f"Spearman ρ (SOD) : {metrics['rho_sod']:.4f}")
    for warning in warnings_log:
        print(f"WARNING: {warning}")

    with open(spearman_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved Spearman metrics to {spearman_metrics_path}")

    if metrics['rho_cod'] < 0.50:
        print(f"\n--- ERROR: LOW_RHO_COD ---")
        print(f"Final COD Spearman correlation ({metrics['rho_cod']:.4f}) is below the 0.50 threshold.")
        exit(1)

    # Generate diagnostic plots
    create_diagnostic_plots(data_df, metrics, output_dir)
    
    # Final Output
    print("\n--- Step 5: Generating final outputs ---")
    output_df = data_df[['img_path', 'mask_path', 'score', 'domain']].copy()
    output_df['img_path'] = output_df['img_path'].astype(str)
    output_df['mask_path'] = output_df['mask_path'].astype(str)
    output_df['score'] = output_df['score'].round(4)
    
    output_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Successfully generated {csv_path}")
    print("--- Pipeline Finished ---")

def get_file_list(dataset_root: Path):
    """
    Enumerates every mask, determines its domain (COD/SOD) and level (L),
    and constructs the path to the corresponding image file.
    """
    mask_dirs = [
        dataset_root / "TrainDataset" / "Mask",
        dataset_root / "TestDataset" / "COD10K" / "Mask",
    ]

    # Mappings from v to L, where lower L is harder/less salient
    cod_v_map = {255: 1, 204: 2, 153: 3, 102: 4, 51: 5}
    sod_v_map = {153: 1, 178: 2, 204: 3, 229: 4, 255: 5}

    all_files = []
    
    print("Enumerating and validating mask files...")
    mask_paths = []
    for d in mask_dirs:
        if d.exists():
            mask_paths.extend(glob.glob(str(d / "*.png")))
        else:
            warnings.warn(f"Mask directory not found: {d}")

    for mask_path_str in tqdm(mask_paths, desc="Processing masks"):
        mask_path = Path(mask_path_str)
        fname = mask_path.stem
        
        parts = fname.rsplit('_', 1)
        if len(parts) != 2 or not parts[1].isdigit():
            warnings.warn(f"Skipping mask with unexpected name format: {mask_path.name}")
            continue
        
        name, v_str = parts
        v = int(v_str)

        domain, level = None, None
        if name.startswith("COD10K") or name.startswith("camourflage"):
            if v in cod_v_map:
                domain = "COD"
                level = cod_v_map[v]
            else:
                warnings.warn(f"Skipping COD mask with unexpected v={v}: {mask_path.name}")
                continue
        elif name.startswith("COCO"):
            if v in sod_v_map:
                domain = "SOD"
                level = sod_v_map[v]
            else:
                warnings.warn(f"Skipping SOD mask with unexpected v={v}: {mask_path.name}")
                continue
        else:
            # This case should ideally not be hit if directory structure is as expected
            warnings.warn(f"Skipping mask with unknown type/prefix: {mask_path.name}")
            continue

        # Construct image path
        img_fname = f"{fname}.jpg"
        img_path = None
        
        # The image could be in one of two corresponding 'Imgs_full' folders
        for d in mask_dirs:
            potential_img_path = d.parent / "Imgs_full" / img_fname
            if potential_img_path.exists():
                img_path = potential_img_path
                break
        
        if img_path is None:
            warnings.warn(f"Image file not found for mask, skipping: {img_fname}")
            continue

        all_files.append({
            "mask_path": mask_path,
            "img_path": img_path,
            "domain": domain,
            "level": level,
            "v": v,
            "img_name_base": name # For grouping SOD masks
        })
        
    return all_files

def build_comparisons(df: pd.DataFrame, seed: int) -> list:
    """Builds the list of (winner, loser) pairs for Bradley-Terry."""
    rng = np.random.default_rng(seed)
    comparisons = []
    
    cod_df = df[df['domain'] == 'COD'].copy()
    sod_df = df[df['domain'] == 'SOD'].copy()

    # A. SOD within-image comparisons
    print("  A. Generating SOD within-image comparisons...")
    sod_df['img_name_base'] = sod_df['mask_path'].apply(lambda p: Path(p).stem.rsplit('_', 1)[0]) # type: ignore
    sod_groups = sod_df.groupby('img_name_base')
    for _, group in tqdm(sod_groups, desc="SOD Groups"):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item1 = group.iloc[i]
                item2 = group.iloc[j]
                if item1['level'] > item2['level']:
                    comparisons.append((item1['mask_id'], item2['mask_id']))
                elif item2['level'] > item1['level']:
                    comparisons.append((item2['mask_id'], item1['mask_id']))

    # A.2. SOD cross-image comparisons (sampled)
    print("  A.2. Generating cross-image SOD comparisons (L5 vs L1/L2)...")
    sod_l5_ids = sod_df[sod_df['level'] == 5]['mask_id'].to_numpy() # type: ignore
    sod_l1_ids = sod_df[sod_df['level'] == 1]['mask_id'].to_numpy() # type: ignore
    sod_l2_ids = sod_df[sod_df['level'] == 2]['mask_id'].to_numpy() # type: ignore
    
    sod_hard_ids = np.concatenate([sod_l1_ids, sod_l2_ids])

    if len(sod_l5_ids) > 0 and len(sod_hard_ids) > 0:
        # Cap sampling to prevent imbalance
        num_to_sample = min(len(sod_l5_ids) * len(sod_hard_ids), 2_000_000)
        print(f"    Sampling {num_to_sample} L5>L1/L2 pairs...")
        
        winner_indices = rng.choice(len(sod_l5_ids), size=num_to_sample, replace=True)
        loser_indices = rng.choice(len(sod_hard_ids), size=num_to_sample, replace=True)

        for w_idx, l_idx in zip(winner_indices, loser_indices):
            comparisons.append((sod_l5_ids[w_idx], sod_hard_ids[l_idx]))

    num_sod_comps = len(comparisons)

    # B. COD cross-dataset comparisons (sampled)
    print("  B. Generating COD cross-dataset comparisons (10% sample)...")
    cod_levels = cod_df.set_index('mask_id')['level']
    
    # Efficiently generate all pairs and sample
    from itertools import combinations
    cod_ids = cod_df['mask_id'].to_numpy() # type: ignore
    
    indices = np.arange(len(cod_ids))
    possible_pairs = np.array(list(combinations(indices, 2)))
    
    level1 = cod_levels.iloc[possible_pairs[:, 0]].values
    level2 = cod_levels.iloc[possible_pairs[:, 1]].values
    valid_pairs_mask = level1 != level2
    valid_pairs = possible_pairs[valid_pairs_mask]
    
    sample_size = int(0.1 * len(valid_pairs))
    sampled_indices = rng.choice(len(valid_pairs), size=sample_size, replace=False)
    sampled_pairs = valid_pairs[sampled_indices]
    
    for i, j in tqdm(sampled_pairs, desc="Sampling COD pairs"):
        id1, id2 = cod_ids[i], cod_ids[j]
        if cod_levels.loc[id1] > cod_levels.loc[id2]:
            comparisons.append((id1, id2))
        else:
            comparisons.append((id2, id1))

    num_cod_comps = len(comparisons) - num_sod_comps

    # C. Cross-domain "soft gap" (SAMPLED)
    print("  C. Generating cross-domain SOD > COD comparisons (sampled)...")
    sod_ids_list = sod_df['mask_id'].tolist()
    cod_ids_list = cod_df['mask_id'].tolist()
    
    if len(sod_ids_list) > 0 and len(cod_ids_list) > 0:
        # Sample a number of pairs comparable to the COD-COD set
        sample_size_cross = min(int(num_cod_comps * 1.2), 2_000_000)

        print(f"    Sampling {sample_size_cross} SOD>COD pairs...")
        
        # Randomly select SOD and COD masks to form pairs
        sampled_sod_indices = rng.choice(len(sod_ids_list), size=sample_size_cross, replace=True)
        sampled_cod_indices = rng.choice(len(cod_ids_list), size=sample_size_cross, replace=True)
        
        for i in range(sample_size_cross):
            sod_id = sod_ids_list[sampled_sod_indices[i]]
            cod_id = cod_ids_list[sampled_cod_indices[i]]
            comparisons.append((sod_id, cod_id))

    return comparisons

def fit_bradley_terry(comparisons: list, all_ids: np.ndarray) -> dict | None:
    """
    Fits a Bradley-Terry-like model using Logistic Regression. This is framed
    as a binary classification problem where for each pair, we predict if the
    outcome is a "win" (1) or a "loss" (0).
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import csc_matrix, vstack

    try:
        id_map = {id_str: i for i, id_str in enumerate(all_ids)}
        num_items = len(all_ids)
        
        # Create a sparse matrix where each row represents a comparison.
        # For a (winner, loser) pair, the winner gets +1 and the loser -1.
        rows, cols, data = [], [], []
        for i, (winner, loser) in enumerate(comparisons):
            rows.append(i); cols.append(id_map.get(winner)); data.append(1)
            rows.append(i); cols.append(id_map.get(loser)); data.append(-1)
        
        X_pos = csc_matrix((data, (rows, cols)), shape=(len(comparisons), num_items))

        # To create a balanced binary classification problem, we augment the data.
        # Every "win" case (y=1) is mirrored by a "loss" case (y=0).
        X_neg = -X_pos
        
        X_full = vstack([X_pos, X_neg])
        y_full = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
        
        # Use a robust solver with a very high C to approximate no penalty,
        # as penalty='none' is not supported by the 'saga' solver.
        # verbose=10 will print progress updates.
        model = LogisticRegression(
            penalty='l2', C=1e8, fit_intercept=False, solver='saga', tol=1e-6, max_iter=100, verbose=10
        )
        model.fit(X_full, y_full)
        
        weights = {id_str: model.coef_[0][i] for id_str, i in id_map.items()}
        return weights

    except Exception as e:
        warnings.warn(f"Logistic Regression model fitting failed: {e}")
        return None

def post_process_weights(df: pd.DataFrame, gamma: float, seed: int) -> tuple[pd.DataFrame, dict]:
    """
    Calibrates weights to a final score, using a balanced subset of data
    to define the normalization functions (ECDF, Isotonic Regression) to avoid
    distribution skew from imbalanced class sizes.
    """
    from scipy.stats import rankdata

    cod_df = df[df['domain'] == 'COD'].copy()
    sod_df = df[df['domain'] == 'SOD'].copy()

    # Create a balanced dataframe for calibration by downsampling the larger class
    if len(cod_df) > len(sod_df):
        calib_cod_df = cod_df.sample(n=len(sod_df), random_state=seed)
        calib_sod_df = sod_df
    else:
        calib_cod_df = cod_df
        calib_sod_df = sod_df.sample(n=len(cod_df), random_state=seed)
    
    calib_df = pd.concat([calib_cod_df, calib_sod_df])
    print(f"Creating calibration functions from a balanced set of {len(calib_df)} samples.")

    # 1. Center weights using the median of the BALANCED set
    calib_median = calib_df['w'].median()
    df['w_centered'] = df['w'] - calib_median
    
    # 2. Build ECDF on the BALANCED set
    ecdf = ECDF(calib_df['w'] - calib_median)
    df['q'] = ecdf(df['w_centered'])
    ecdf_params = {'x': ecdf.x.tolist(), 'y': ecdf.y.tolist()}
    
    # 3. Optional gamma-correction
    df['q_gamma'] = df['q'] ** gamma
    
    # 4. Fit IsotonicRegression on the BALANCED set
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    
    # Get quantiles for the balanced set to fit the isotonic model
    calib_q_gamma = ecdf(calib_df['w'] - calib_median) ** gamma
    y_target = rankdata(calib_q_gamma, method='average') / len(calib_q_gamma)
    
    iso.fit(calib_q_gamma, y_target)
    
    # 5. Transform the FULL dataset using the fitted model
    df['z'] = iso.transform(df['q_gamma'])
    
    return df, ecdf_params

def calculate_diagnostics(df: pd.DataFrame) -> tuple[dict, list]:
    """Calculates Spearman correlation and uniformity metrics."""
    from scipy.stats import entropy

    warnings_log = []
    
    # --- Spearman Metrics ---
    cod_df = df[df['domain'] == 'COD']
    sod_df = df[df['domain'] == 'SOD']
    
    # 1. rho_COD
    rho_cod_res = spearmanr(cod_df['score'], cod_df['level'])
    rho_cod = rho_cod_res.correlation if hasattr(rho_cod_res, 'correlation') else 0.0 # type: ignore

    # 2. rho_SOD (mean per-image)
    sod_image_rhos = []
    sod_groups = sod_df.groupby('img_name_base')
    for _, grp in sod_groups:
        if pd.Series(grp['level']).nunique() > 1:
            res = spearmanr(grp['score'], grp['level'])
            if hasattr(res, 'correlation') and not np.isnan(res.correlation):
                sod_image_rhos.append(res.correlation) # type: ignore
    rho_sod = np.nanmean(sod_image_rhos) if sod_image_rhos else 0.0

    metrics = {"rho_cod": rho_cod, "rho_sod": rho_sod}

    # --- Uniformity Check (KL Divergence) ---
    # COD scores in [0, 5]
    cod_scores = cod_df['score']
    hist_cod, _ = np.histogram(cod_scores, bins=25, range=(0, 5), density=True)
    if hist_cod.sum() > 0:
        hist_cod = hist_cod / hist_cod.sum() # Normalize to probability distribution
        kl_cod = entropy(hist_cod, qk=np.ones(25)/25)
    else:
        kl_cod = np.inf
    metrics['kl_div_cod'] = kl_cod
    if kl_cod > 0.10:
        warnings_log.append(f"NON-UNIFORM (COD): KL divergence is {kl_cod:.4f} > 0.10")

    # SOD scores in [5, 10]
    sod_scores = sod_df['score']
    hist_sod, _ = np.histogram(sod_scores, bins=25, range=(5, 10), density=True)
    if hist_sod.sum() > 0:
        hist_sod = hist_sod / hist_sod.sum() # Normalize
        kl_sod = entropy(hist_sod, qk=np.ones(25)/25)
    else:
        kl_sod = np.inf
    metrics['kl_div_sod'] = kl_sod
    if kl_sod > 0.10:
        warnings_log.append(f"NON-UNIFORM (SOD): KL divergence is {kl_sod:.4f} > 0.10")
        
    return metrics, warnings_log

def create_diagnostic_plots(df: pd.DataFrame, metrics: dict, output_dir: Path):
    """Generates and saves a figure with diagnostic plots."""
    print("Generating diagnostic plots...")
    import matplotlib.pyplot as plt
    
    cod_df = df[df['domain'] == 'COD']
    sod_df = df[df['domain'] == 'SOD']

    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(2, 2)
    fig.suptitle("Score Diagnostics (Scheme B-soft)")

    # --- Subplot A: Score vs Level (Boxplot) ---
    ax1 = fig.add_subplot(gs[0, :])
    all_levels = sorted(pd.Series(df['level']).unique())
    data_to_plot = [df[df['level'] == lvl]['score'] for lvl in all_levels]
    
    # Separate COD and SOD levels for coloring
    cod_levels = sorted(pd.Series(cod_df['level']).unique())
    
    bp = ax1.boxplot(data_to_plot, patch_artist=True)
    ax1.set_xticklabels(all_levels)
    
    for i, patch in enumerate(bp['boxes']):
        level = all_levels[i]
        if level in cod_levels:
            patch.set_facecolor('lightblue') # COD
        else:
            patch.set_facecolor('lightsalmon') # SOD
            
    ax1.set_xlabel("Discrete Level (L)")
    ax1.set_ylabel("Final Score s")
    ax1.set_title("A: Final Score vs. Discrete Level")
    
    text_str = (
        f"$\\rho_{{COD}}$: {metrics['rho_cod']:.3f}\n"
        f"$\\rho_{{SOD}}$: {metrics['rho_sod']:.3f} (mean per-image)"
    )
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # --- Subplot B: Score Distribution (Histogram) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(cod_df['score'], bins=25, range=(0, 5), density=True, alpha=0.7, label=f'COD (KL={metrics["kl_div_cod"]:.3f})', color='blue')
    ax2.hist(sod_df['score'], bins=25, range=(5, 10), density=True, alpha=0.7, label=f'SOD (KL={metrics["kl_div_sod"]:.3f})', color='red')
    ax2.axhline(y=1/5, xmin=0.0, xmax=0.5, color='gray', linestyle='--', label='Uniform (COD)')
    ax2.axhline(y=1/5, xmin=0.5, xmax=1.0, color='gray', linestyle='--')
    ax2.set_xlabel("Final Score s")
    ax2.set_ylabel("Density")
    ax2.set_title("B: Score Distribution by Domain")
    ax2.legend()

    # --- Subplot C: z vs. q_gamma (Isotonic Fit) ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df['q_gamma'], df['z'], s=1, alpha=0.1, label='Fitted z_i')
    ax3.plot([0, 1], [0, 1], 'r--', label='Ideal y=x')
    ax3.set_xlabel("Corrected Quantile q_i^γ")
    ax3.set_ylabel("Isotonic Output z_i")
    ax3.set_title("C: Isotonic Regression Mapping")
    ax3.legend()
    ax3.grid(True, linestyle=':')

    # --- Save Figure ---
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = output_dir / "score_diagnostics.png"
    pdf_path = output_dir / "score_diagnostics.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Saved diagnostic plots to {png_path} and {pdf_path}")


if __name__ == '__main__':
    main() 