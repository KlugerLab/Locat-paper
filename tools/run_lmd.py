# run_lmd.py
# Robust LMD wrapper with explicit R-matrix construction, reliable dimnames,
# and an R-side validator that aligns cells, drops zero-var PCs, enforces min_cell, & adjusts knn.

import numpy as np
import pandas as pd
import anndata as ad
import numpy as np
from scipy import sparse
import numpy as np
from scipy import sparse


def prep_W_from_adata_connectivities(
    adata,
    k: int | None = None,        # keep top-k neighbors per row (excluding self); None = keep all
    binarize: bool = False,      # True -> set nonzeros to 1
    add_self_loops: bool = True, # ensure 1.0 on the diagonal
) -> sparse.csr_matrix:
    if "connectivities" not in adata.obsp:
        raise KeyError("adata.obsp['connectivities'] not found.")
    W = adata.obsp["connectivities"]
    W = W.tocsr().astype(np.float64, copy=False)

    # symmetrize (Scanpy's is usually symmetric, but be safe)
    W = W.maximum(W.T)

    # optional binarization
    if binarize:
        W.data[:] = 1.0

    # optional top-k pruning per row (keep diag separately)
    if k is not None:
        W = W.tolil(copy=True)
        n = W.shape[0]
        for i in range(n):
            row = W.rows[i]
            data = W.data[i]
            if len(row) <= k + 1:   # +1 for possible self
                continue
            # ensure self-loop index present for sorting
            if i not in row:
                row.append(i); data.append(0.0)
            # sort by weight desc, keep top k+1 (incl. self)
            order = np.argsort(-np.array(data))
            keep_idx = set(np.array(row)[order[:k+1]].tolist())
            # prune
            new_row, new_data = [], []
            for rj, dj in zip(row, data):
                if rj in keep_idx:
                    new_row.append(rj); new_data.append(dj)
            W.rows[i] = new_row
            W.data[i] = new_data
        W = W.tocsr()

    # add/ensure self-loops
    if add_self_loops:
        diag = sparse.diags(np.ones(W.shape[0], dtype=np.float64), format="csr")
        W = W.maximum(diag)

    W.eliminate_zeros()
    return W

from rpy2 import robjects as ro
from rpy2.robjects import conversion as rconv, default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

# ---- rpy2 global converter (kept, but we use a plain default_converter block for R object creation) ----
rconv.set_conversion(default_converter + numpy2ri.converter)

# ---- Import the R package once ----
LMD = importr("LocalizedMarkerDetector")

# ---- R validator/runner: define once at import time ----
ro.r("""
validate_and_run <- function(dat, feature_space, knn=5L, max_time=NA_integer_, min_cell=5L, verbose=TRUE) {
  # Basic structure checks
  if (!is.matrix(dat))            stop("`dat` must be a matrix (genes x cells).")
  if (!is.matrix(feature_space))  stop("`feature_space` must be a matrix (cells x dims).")
  if (is.null(rownames(dat)) || is.null(colnames(dat)))
    stop("`dat` must have rownames (genes) and colnames (cells).")
  if (is.null(rownames(feature_space)))
    stop("`feature_space` must have rownames (cells).")

  # Types & finiteness
  storage.mode(dat) <- "double"
  storage.mode(feature_space) <- "double"
  if (any(!is.finite(dat)))            stop("Non-finite values in `dat`.")
  if (any(!is.finite(feature_space)))  stop("Non-finite values in `feature_space`.")

  # Duplicate IDs
  if (any(duplicated(colnames(dat))))
    stop("Duplicate cell IDs in `dat` colnames.")
  if (any(duplicated(rownames(feature_space))))
    stop("Duplicate cell IDs in `feature_space` rownames.")

  # Align cells (columns of dat vs rownames of feature_space)
  if (!identical(colnames(dat), rownames(feature_space))) {
    common <- intersect(colnames(dat), rownames(feature_space))
    if (length(common) < 2L)
      stop("Too few overlapping cell IDs between `dat` and `feature_space`.")
    dat  <- dat[, common, drop=FALSE]
    feature_space <- feature_space[common, , drop=FALSE]
    if (verbose) message("Aligned to common cells: ", length(common))
  }

  # Drop zero-variance PC columns (rare but fatal)
  if (ncol(feature_space) == 0L) stop("`feature_space` has 0 columns.")
  v <- apply(feature_space, 2L, stats::var)
  if (any(v == 0)) {
    keep <- which(v > 0)
    if (length(keep) == 0L) stop("All PC dimensions have zero variance after alignment.")
    feature_space <- feature_space[, keep, drop=FALSE]
    if (verbose) message("Dropped zero-variance PC dims: ", sum(v == 0))
  }

  # Enforce min_cell **by gene row** (LMD expects genes in rows)
  nz <- rowSums(dat > 0)
  keepg <- which(nz >= as.integer(min_cell))
  if (length(keepg) == 0L) stop("After min_cell filtering, no genes remain.")
  if (length(keepg) < nrow(dat)) {
    if (verbose) message("Filtered genes by min_cell: kept ", length(keepg), " / ", nrow(dat))
    dat <- dat[keepg, , drop=FALSE]
  }

  # knn sanity
  ncell <- ncol(dat)
  if (knn >= ncell) {
    knn <- max(1L, ncell - 1L)
    if (verbose) message("Adjusted knn to ", knn, " for ncell = ", ncell)
  }

  # Final informative prints
  if (verbose) {
    message(sprintf("Final dims: dat (genes x cells) = %d x %d; feature_space (cells x dims) = %d x %d",
                    nrow(dat), ncol(dat), nrow(feature_space), ncol(feature_space)))
    message("Names aligned (cells): ", isTRUE(all.equal(colnames(dat), rownames(feature_space))))
  }

  args <- list(expression=dat, feature_space=feature_space, knn=as.integer(knn))
  if (!is.na(max_time)) args$max_time <- as.integer(max_time)

  # Run LMD
  do.call(LocalizedMarkerDetector::LMD, args)
}
""")
_validate_and_run = ro.globalenv["validate_and_run"]

from rpy2.robjects.packages import importr
Matrix = importr("Matrix")

def _extract_affinity_from_cg(cg):
    """
    Accepts the return object from LocalizedMarkerDetector::CoarseGrain(...)
    and returns an R Matrix::dgCMatrix adjacency with dimnames.
    Handles variants where the graph is stored as:
      - 'graph' (igraph), or
      - 'graph.affinity' (sparse matrix), or
      - other common aliases.
    """
    # what slots exist?
    cg_names = list(ro.r("names")(cg))
    # try common slots in order
    candidates = ["graph.affinity", "graph", "W", "affinity", "A"]
    name = next((n for n in candidates if n in cg_names), None)
    if name is None:
        raise RuntimeError(f"CoarseGrain() returned unexpected names: {cg_names}")

    obj = cg.rx2(name)

    # If it's already a sparse Matrix, just coerce to dgCMatrix
    klass = list(ro.r("class")(obj))
    if any(k.startswith("dg") and k.endswith("Matrix") for k in klass) or "sparseMatrix" in klass:
        return ro.r["as"](obj, "dgCMatrix")

    # If it's an igraph, convert to adjacency
    if "igraph" in klass:
        # require igraph and Matrix
        importr("igraph")
        importr("Matrix")
        # weighted adjacency as sparse
        adj = ro.r("function(g) igraph::as_adjacency_matrix(g, type='both', attr='weight', sparse=TRUE)")(obj)
        adj = ro.r["as"](adj, "dgCMatrix")
        # ensure symmetric + self-loops
        adj = ro.r("function(M){ M <- Matrix::forceSymmetric(M, uplo='U'); Matrix::diag(M) <- 1; Matrix::as(M,'dgCMatrix') }")(adj)
        return adj

    # If it’s something else (e.g., dense matrix), force to sparse
    importr("Matrix")
    if "matrix" in klass or "array" in klass:
        adj = ro.r["Matrix"](obj, sparse=True)
        adj = ro.r["as"](adj, "dgCMatrix")
        adj = ro.r("function(M){ M <- Matrix::forceSymmetric(M, uplo='U'); Matrix::diag(M) <- 1; Matrix::as(M,'dgCMatrix') }")(adj)
        return adj

    raise RuntimeError(f"Unsupported coarse graph class: {klass}")


def _csr_to_R_dgCMatrix(W_csr, names):
    """
    Convert scipy CSR/COO to R Matrix::dgCMatrix and set dimnames safely.
    """
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter

    W = W_csr.tocoo(copy=False)
    n = W.shape[0]

    with localconverter(default_converter):
        r_i = ro.IntVector((W.row.astype(np.int64) + 1).tolist())  # 1-based
        r_j = ro.IntVector((W.col.astype(np.int64) + 1).tolist())
        r_x = ro.FloatVector(W.data.astype(np.float64).tolist())
        r_dims = ro.IntVector([n, n])

        # build sparse matrix first (no dimnames)
        r_mat = Matrix.sparseMatrix(i=r_i, j=r_j, x=r_x, dims=r_dims)
        r_mat = ro.r['as'](r_mat, "dgCMatrix")

        # now assign dimnames using an unnamed R list
        r_dimnames = ro.r.list(StrVector(names), StrVector(names))
        r_mat = ro.r['dimnames<-'](r_mat, r_dimnames)

    return r_mat




def lmd_scores_coarse_from_adata_preknn(
    adata: ad.AnnData,
    feature_space_key: str = "X_pca",
    W_csr: sparse.csr_matrix | None = None,   # <- required if we skip R KNN
    min_cells: int = 5,
    coarse_N: int = 10000,
    max_time: int = 2**12,
    assume_counts: bool = False,
    center_scale_pcs: bool = False,
    debug: bool = False,
) -> pd.Series:
    # ----- expression (genes x cells) -----
    X = adata.X if hasattr(adata, "X") else adata.layers[None]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    det = (X > 0).sum(axis=0)
    keep = det >= int(min_cells)
    if keep.sum() == 0:
        raise ValueError("No genes remain after min_cells filtering.")
    X = X[:, keep]
    X = np.log1p(X, dtype=np.float64) if assume_counts else X.astype(np.float64, copy=False)

    genes = np.asarray(adata.var_names)[keep]
    cells = np.asarray(adata.obs_names, dtype=object)
    dat = X.T  # genes x cells

    # ----- feature space -----
    feat = _get_feat(adata, feature_space_key).astype(np.float64, copy=False)
    if feat.shape[0] != len(cells):
        raise ValueError(f"Feature space rows ({feat.shape[0]}) != n_cells ({len(cells)}).")
    if center_scale_pcs:
        mean = feat.mean(axis=0, keepdims=True)
        std = np.clip(feat.std(axis=0, ddof=1, keepdims=True), 1e-12, None)
        feat = (feat - mean) / std

    # ----- R conversion -----
    genes_py = [f"gene_{i+1}" if (g is None or str(g) == "") else str(g) for i, g in enumerate(genes)]
    cells_py = [f"cell_{i+1}" if (c is None or str(c) == "") else str(c) for i, c in enumerate(cells)]

    with localconverter(default_converter):
        vec_dat  = ro.FloatVector(dat.ravel(order="F"))
        vec_feat = ro.FloatVector(feat.ravel(order="F"))
        r_dat  = ro.r.matrix(vec_dat,  nrow=dat.shape[0],  ncol=dat.shape[1])   # genes x cells
        r_feat = ro.r.matrix(vec_feat, nrow=feat.shape[0], ncol=feat.shape[1])  # cells x dims
        r_dat  = ro.r['storage.mode<-'](r_dat,  "double")
        r_feat = ro.r['storage.mode<-'](r_feat, "double")
        r_dat  = ro.r['rownames<-'](r_dat,  StrVector(genes_py))
        r_dat  = ro.r['colnames<-'](r_dat,  StrVector(cells_py))
        r_feat = ro.r['rownames<-'](r_feat, StrVector(cells_py))
        r_dat  = ro.r['rownames<-'](r_dat, ro.r('make.unique')(ro.r('rownames')(r_dat)))
        r_dat  = ro.r['colnames<-'](r_dat, ro.r('make.unique')(ro.r('colnames')(r_dat)))
        r_feat = ro.r['rownames<-'](r_feat, ro.r('make.unique')(ro.r('rownames')(r_feat)))

        if debug:
            nrg = int(ro.r['nrow'](r_dat)[0]);   ncg = int(ro.r['ncol'](r_dat)[0])
            nrc = int(ro.r['nrow'](r_feat)[0]); ncc = int(ro.r['ncol'](r_feat)[0])
            print(f"R dims dat (genes x cells): {nrg} x {ncg}, feature_space: {nrc} x {ncc}")
            print("Passing precomputed kNN graph from Python ...")

        if W_csr is None:
            raise ValueError("W_csr (precomputed kNN graph) is required for the *_preknn function.")

        # convert CSR -> dgCMatrix with dimnames
        W = _csr_to_R_dgCMatrix(W_csr, list(cells_py))

        # ----- Coarse-grain -----
        CoarseGrain = _r_get("LocalizedMarkerDetector", "CoarseGrain")
        cg = CoarseGrain(
            feature_space=r_feat,
            expression=r_dat,
            **{'graph.affinity': W},
            N=int(coarse_N),
            random_seed=1
        )
        cg_expr = cg.rx2("expression")
        if cg_expr is None:
            raise RuntimeError(f"Did not find 'expression' in CoarseGrain() output. Names: {list(ro.r('names')(cg))}")
        
        # robustly get a dgCMatrix adjacency
        cg_graph = _extract_affinity_from_cg(cg)
        
        if debug:
            print("Coarse graph class:", list(ro.r("class")(cg_graph)))
            print("Coarse graph dims:", int(ro.r["nrow"](cg_graph)[0]), "x", int(ro.r["ncol"](cg_graph)[0]))


        # ----- Diffusion operators (with dense fallback) -----
        ConstructDiffusionOperators = _r_get("LocalizedMarkerDetector", "ConstructDiffusionOperators")
        try:
            P_ls = ConstructDiffusionOperators(W=cg_graph, max_time=int(max_time))
        except Exception as e:
            if debug:
                print("ConstructDiffusionOperators failed with sparse matrix; retrying with dense base matrix via as.matrix ...")
            W_dense_for_ops = ro.r['as.matrix'](cg_graph)
            P_ls = ConstructDiffusionOperators(W=W_dense_for_ops, max_time=int(max_time))
        
        if debug:
            print("Converting diffusion operators to sparse matrices...")
        
        # ----- Normalize init state on coarse graph columns -----
        RowwiseNormalize = _r_get("LocalizedMarkerDetector", "RowwiseNormalize")
        cols = ro.r["colnames"](cg_graph)
        rho = RowwiseNormalize(cg_expr.rx(True, cols))
        
        # Helper to coerce to base dense matrix
        as_matrix = ro.r['as.matrix']
        lapply = ro.r['lapply']
        
        # Always run dense to avoid t.default() on S4s
        W_run   = as_matrix(cg_graph)
        P_ls_run = lapply(P_ls, as_matrix)
        rho_run  = as_matrix(rho)
        
        fast_get_lmds = _r_get("LocalizedMarkerDetector", "fast_get_lmds")
        
        if debug:
            print("Max diffusion time:", int(max_time))
            print("Calculate LMD score profile for large data...")
        
        # Single call — do NOT call fast_get_lmds again elsewhere.
        res = fast_get_lmds(W=W_run, init_state=rho_run, P_ls=P_ls_run, largeData=True, highres=False)

        # ----- scores -----
        res_names = list(ro.r('names')(res))
        slot = next((s for s in ["cumulative_score","cumulative_scores","scores","score"] if s in res_names), None)
        if slot is None:
            raise RuntimeError(f"Could not find score slot in result. Available: {res_names}")
        scores = np.asarray(res.rx2(slot), dtype=np.float64)

    if scores.ndim != 1 or scores.shape[0] != genes.shape[0]:
        raise ValueError(f"Bad scores shape {scores.shape} for {genes.shape[0]} genes.")
    return pd.Series(scores, index=genes, name="LMD_score")


def _get_feat(adata: ad.AnnData, key: str) -> np.ndarray:
    """
    Fetch feature space from .obsm; tries 'pca' <-> 'X_pca' fallback.
    Returns a (cells x dims) float64 ndarray.
    """
    if key in adata.obsm:
        arr = adata.obsm[key]
    else:
        alt = "X_pca" if key == "pca" else "pca"
        if alt in adata.obsm:
            arr = adata.obsm[alt]
        else:
            raise KeyError(
                f"Feature space '{key}' not in adata.obsm; fallback '{alt}' also missing."
            )
    arr = arr.A if hasattr(arr, "A") else np.asarray(arr)
    return np.asarray(arr, dtype=np.float64)


def lmd_scores_from_adata(
    adata: ad.AnnData,
    X_key: str | None = None,             # None -> adata.X
    feature_space_key: str = "X_pca",     # or "pca"
    min_cells: int = 5,
    max_time: int | None = None,          # let LMD choose if None
    knn: int = 5,                         # tutorial default
    assume_counts: bool = False,          # True if X are raw counts (to log1p)
    center_scale_pcs: bool = False,       # optional: center/scale PCs before LMD
    normalize_names: bool = False,        # make obs/var names str() and unique
    debug: bool = False,
) -> pd.Series:
    """
    Compute LMD per-gene 'cumulative_score' using the one-step LMD(dat, feature_space, ...),
    matching the tutorial's expectations:

      - dat : genes x cells (rownames = genes, colnames = cells; preferably log-normalized)
      - feature_space : cells x dims (rownames = same cell IDs and order)

    Returns
    -------
    pandas.Series of length = #kept genes, indexed by gene names.
    """

    # Optional normalization of names to avoid weird bytes/duplicates
    if normalize_names:
        adata.obs_names = adata.obs_names.astype(str)
        adata.var_names = adata.var_names.astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()

    # ----- 1) Expression (cells x genes) -> filter -> optional log1p -> TRANSPOSE to genes x cells
    X = adata.layers[X_key] if X_key else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if X.ndim != 2:
        raise ValueError("adata.X (or selected layer) must be 2D (cells x genes).")

    det = (X > 0).sum(axis=0)  # per gene
    keep = det >= int(min_cells)
    if keep.sum() == 0:
        raise ValueError("After filtering, no genes remain (increase min_cells or check data).")

    X = X[:, keep]
    if assume_counts:
        X = np.log1p(X, dtype=np.float64)
    else:
        X = X.astype(np.float64, copy=False)  # assume already log-normalized

    genes = np.asarray(adata.var_names)[keep]    # kept genes
    cells = np.asarray(adata.obs_names, dtype=object)

    # TRANSPOSE here: genes x cells
    dat = X.T  # shape: (n_genes_kept, n_cells)

    # ----- 2) Feature space (cells x dims)
    feat = _get_feat(adata, feature_space_key).astype(np.float64, copy=False)
    if feat.shape[0] != len(cells):
        raise ValueError(
            f"Feature space rows ({feat.shape[0]}) must equal number of cells ({len(cells)})."
        )

    if center_scale_pcs:
        # center/scale each PC column (avoid zero-variance crashes downstream)
        mean = feat.mean(axis=0, keepdims=True)
        std = np.clip(feat.std(axis=0, ddof=1, keepdims=True), 1e-12, None)
        feat = (feat - mean) / std

    # ----- 3) Build R matrices inside a plain conversion context (no numpy2ri)
    # This avoids the missing-conversion error and prevents auto NumPy casting that can drop dimnames.
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter

    # Clean names (force non-empty strings so R will keep them)
    genes_py = [("" if g is None else str(g)) for g in genes]
    cells_py = [("" if c is None else str(c)) for c in cells]
    if any(g == "" for g in genes_py):
        genes_py = [f"gene_{i+1}" if g == "" else g for i, g in enumerate(genes_py)]
    if any(c == "" for c in cells_py):
        cells_py = [f"cell_{i+1}" if c == "" else c for i, c in enumerate(cells_py)]

    # Construct R matrices and set dimnames explicitly
    with localconverter(default_converter):
        # R numeric vectors (already column-major flattening to match R)
        vec_dat  = ro.FloatVector(dat.ravel(order="F"))
        vec_feat = ro.FloatVector(feat.ravel(order="F"))

        # R matrices
        r_dat  = ro.r.matrix(vec_dat,  nrow=dat.shape[0],  ncol=dat.shape[1])
        r_feat = ro.r.matrix(vec_feat, nrow=feat.shape[0], ncol=feat.shape[1])

        # Ensure double
        r_dat  = ro.r['storage.mode<-'](r_dat,  "double")
        r_feat = ro.r['storage.mode<-'](r_feat, "double")

        # Assign names SEPARATELY (reassign the returned objects!)
        r_dat  = ro.r['rownames<-'](r_dat,  StrVector(genes_py))   # rows = genes
        r_dat  = ro.r['colnames<-'](r_dat,  StrVector(cells_py))   # cols = cells
        r_feat = ro.r['rownames<-'](r_feat, StrVector(cells_py))   # rows = cells

        if debug:
            print("Pre-validate R checks:")
            has_rn_dat  = bool(ro.r('function(x) !is.null(rownames(x))')(r_dat)[0])
            has_cn_dat  = bool(ro.r('function(x) !is.null(colnames(x))')(r_dat)[0])
            has_rn_feat = bool(ro.r('function(x) !is.null(rownames(x))')(r_feat)[0])
            print("  has rownames(dat)?", has_rn_dat)
            print("  has colnames(dat)?", has_cn_dat)
            print("  has rownames(feature_space)?", has_rn_feat)

        # Force simple unique fallbacks only if missing
        if not bool(ro.r('function(x) !is.null(rownames(x))')(r_dat)[0]):
            r_dat = ro.r('`rownames<-`')(r_dat, StrVector([f"gene_{i+1}" for i in range(dat.shape[0])]))
        if not bool(ro.r('function(x) !is.null(colnames(x))')(r_dat)[0]):
            r_dat = ro.r('`colnames<-`')(r_dat, StrVector([f"cell_{i+1}" for i in range(dat.shape[1])]))
        if not bool(ro.r('function(x) !is.null(rownames(x))')(r_feat)[0]):
            r_feat = ro.r('`rownames<-`')(r_feat, StrVector([f"cell_{i+1}" for i in range(feat.shape[0])]))

        # Make names unique (guard each call)
        if bool(ro.r('function(x) !is.null(rownames(x))')(r_dat)[0]):
            r_dat = ro.r('`rownames<-`')(r_dat, ro.r('make.unique')(ro.r('rownames')(r_dat)))
        if bool(ro.r('function(x) !is.null(colnames(x))')(r_dat)[0]):
            r_dat = ro.r('`colnames<-`')(r_dat, ro.r('make.unique')(ro.r('colnames')(r_dat)))
        if bool(ro.r('function(x) !is.null(rownames(x))')(r_feat)[0]):
            r_feat = ro.r('`rownames<-`')(r_feat, ro.r('make.unique')(ro.r('rownames')(r_feat)))

        if debug:
            same = bool(ro.r('function(d,f) identical(colnames(d), rownames(f))')(r_dat, r_feat)[0])
            print("Names aligned (cells):", same)
            nrg = int(ro.r['nrow'](r_dat)[0]);   ncg = int(ro.r['ncol'](r_dat)[0])
            nrc = int(ro.r['nrow'](r_feat)[0]); ncc = int(ro.r['ncol'](r_feat)[0])
            print(f"R dims dat (genes x cells): {nrg} x {ncg}, feature_space: {nrc} x {ncc}")

        # ---- Call our R validator/runner (aligns/intersects cells, drops zero-var PCs, etc.)
        res = _validate_and_run(
            r_dat,
            r_feat,
            knn=int(knn),
            max_time=(int(max_time) if max_time is not None else ro.NA_Integer),
            min_cell=int(min_cells),
            verbose=True if debug else False
        )

        # Extract scores
        scores = np.array(res.rx2("cumulative_score"))

    return pd.Series(scores, index=genes, name="LMD_score")

# Put this once near the top of run_lmd.py (after `LMD = importr("LocalizedMarkerDetector")`)
def _r_get(pkg: str, fun: str):
    """Return an R function `pkg::fun`, preferring the imported package object."""
    try:
        pkg_mod = importr(pkg)
        return getattr(pkg_mod, fun)
    except AttributeError:
        return ro.r['::'](pkg, fun)


def lmd_scores_coarse_from_adata(
    adata: ad.AnnData,
    feature_space_key: str = "X_pca",
    min_cells: int = 5,
    knn: int = 5,
    coarse_N: int = 10000,
    max_time: int = 2**12,
    assume_counts: bool = False,
    center_scale_pcs: bool = False,
    normalize_names: bool = False,
    debug: bool = False,
) -> pd.Series:
    # ----- 1) Prepare expression (genes x cells) -----
    X = adata.X if hasattr(adata, "X") else adata.layers[None]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if X.ndim != 2:
        raise ValueError("adata.X must be 2D (cells x genes).")

    det = (X > 0).sum(axis=0)
    keep = det >= int(min_cells)
    if keep.sum() == 0:
        raise ValueError("No genes remain after min_cells filtering.")
    X = X[:, keep]

    if assume_counts:
        X = np.log1p(X, dtype=np.float64)
    else:
        X = X.astype(np.float64, copy=False)

    genes = np.asarray(adata.var_names)[keep]
    cells = np.asarray(adata.obs_names, dtype=object)
    dat = X.T  # genes x cells

    # ----- 2) Feature space (cells x dims) -----
    feat = _get_feat(adata, feature_space_key).astype(np.float64, copy=False)
    if feat.shape[0] != len(cells):
        raise ValueError(f"Feature space rows ({feat.shape[0]}) != n_cells ({len(cells)}).")

    if center_scale_pcs:
        mean = feat.mean(axis=0, keepdims=True)
        std = np.clip(feat.std(axis=0, ddof=1, keepdims=True), 1e-12, None)
        feat = (feat - mean) / std

    # ----- 3) R matrix conversion + dimnames -----
    genes_py = [f"gene_{i+1}" if (g is None or str(g) == "") else str(g) for i, g in enumerate(genes)]
    cells_py = [f"cell_{i+1}" if (c is None or str(c) == "") else str(c) for i, c in enumerate(cells)]

    with localconverter(default_converter):
        # Create R numeric vectors (column-major flattening for R)
        vec_dat  = ro.FloatVector(dat.ravel(order="F"))
        vec_feat = ro.FloatVector(feat.ravel(order="F"))

        # Create R matrices
        r_dat  = ro.r.matrix(vec_dat,  nrow=dat.shape[0],  ncol=dat.shape[1])   # genes x cells
        r_feat = ro.r.matrix(vec_feat, nrow=feat.shape[0], ncol=feat.shape[1])  # cells x dims

        # Ensure double
        r_dat  = ro.r['storage.mode<-'](r_dat,  "double")
        r_feat = ro.r['storage.mode<-'](r_feat, "double")

        # Assign dimnames and ensure uniqueness
        r_dat  = ro.r['rownames<-'](r_dat,  StrVector(genes_py))
        r_dat  = ro.r['colnames<-'](r_dat,  StrVector(cells_py))
        r_feat = ro.r['rownames<-'](r_feat, StrVector(cells_py))

        r_dat  = ro.r['rownames<-'](r_dat, ro.r('make.unique')(ro.r('rownames')(r_dat)))
        r_dat  = ro.r['colnames<-'](r_dat, ro.r('make.unique')(ro.r('colnames')(r_dat)))
        r_feat = ro.r['rownames<-'](r_feat, ro.r('make.unique')(ro.r('rownames')(r_feat)))

        if debug:
            nrg = int(ro.r['nrow'](r_dat)[0]);   ncg = int(ro.r['ncol'](r_dat)[0])
            nrc = int(ro.r['nrow'](r_feat)[0]); ncc = int(ro.r['ncol'](r_feat)[0])
            print(f"R dims dat (genes x cells): {nrg} x {ncg}, feature_space: {nrc} x {ncc}")
            print("Constructing KNN graph in R ...")

        # ----- 4) Construct kNN graph -----
        ConstructKnnGraph = _r_get("LocalizedMarkerDetector", "ConstructKnnGraph")
        knn_res = ConstructKnnGraph(
            knn=int(knn),
            feature_space=r_feat
        )
        W = knn_res.rx2("graph")
        if W is None:
            W = knn_res.rx2("graph.affinity")
        if W is None:
            raise RuntimeError(f"Unexpected KNN result names: {list(ro.r('names')(knn_res))}")

        # ----- 5) Coarse-grain -----
        if debug:
            print(f"Coarse-graining to N={int(coarse_N)} groups ...")

        CoarseGrain = _r_get("LocalizedMarkerDetector", "CoarseGrain")
        cg = CoarseGrain(
            feature_space=r_feat,
            expression=r_dat,
            **{'graph.affinity': W},
            N=int(coarse_N),
            random_seed=1
        )
        cg_expr  = cg.rx2("expression")  # genes x groups
        cg_graph = cg.rx2("graph")       # groups x groups affinity
        if cg_expr is None or cg_graph is None:
            raise RuntimeError(f"Unexpected CoarseGrain result names: {list(ro.r('names')(cg))}")

        # ----- 6) Diffusion operators -----
        if debug:
            print(f"Constructing diffusion operators (max_time={int(max_time)}) ...")

        ConstructDiffusionOperators = _r_get("LocalizedMarkerDetector", "ConstructDiffusionOperators")
        P_ls = ConstructDiffusionOperators(
            W=cg_graph,
            max_time=int(max_time)
        )

        # ----- 7) Normalize & run large-data LMD -----
        RowwiseNormalize = _r_get("LocalizedMarkerDetector", "RowwiseNormalize")
        rho = RowwiseNormalize(
            cg_expr.rx(True, ro.r['colnames'](cg_graph))
        )

        fast_get_lmds = _r_get("LocalizedMarkerDetector", "fast_get_lmds")
        res = fast_get_lmds(
            W=cg_graph,
            init_state=rho,
            P_ls=P_ls,
            largeData=True,
            highres=False
        )

        # ----- 8) Robust score extraction -----
        res_names = list(ro.r('names')(res))
        if debug:
            print("Result names from fast_get_lmds:", res_names)

        slot_candidates = ["cumulative_score", "cumulative_scores", "scores", "score"]
        slot = next((s for s in slot_candidates if s in res_names), None)
        if slot is None:
            raise RuntimeError(f"Could not find score slot in result. Available: {res_names}")
        r_scores = res.rx2(slot)
        if r_scores is None:
            raise RuntimeError(f"Slot '{slot}' is NULL in result.")

        scores = np.asarray(r_scores, dtype=np.float64)

    # ----- 9) Sanity checks & return -----
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {scores.shape}")
    if scores.shape[0] != genes.shape[0]:
        raise ValueError(f"Score length {scores.shape[0]} != #genes kept {genes.shape[0]}")

    return pd.Series(scores, index=genes, name="LMD_score")




# Optional quick smoke test
if __name__ == "__main__":
    n_cells, n_genes, dims = 200, 100, 10
    rng = np.random.default_rng(0)
    X = rng.negative_binomial(2, 0.5, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i:04d}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, dims)).astype(np.float32)

    s = lmd_scores_from_adata(
        adata,
        feature_space_key="X_pca",
        min_cells=5,
        knn=5,
        max_time=None,
        assume_counts=True,      # because X above are counts-like
        center_scale_pcs=True,
        normalize_names=True,
        debug=True
    )
    print(s.head())
