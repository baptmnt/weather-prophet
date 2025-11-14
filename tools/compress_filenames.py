import json
from pathlib import Path
import re
import shutil

ROOT = Path("saved_models")

def extract_hparams(json_data: dict) -> dict:
    return json_data.get("hyperparameters", {})

def build_new_name(hparams: dict, old_model_name: str) -> str:
    def lr_compact(v):
        try:
            return f"lr{v:.0e}".replace('+', '').replace('-', '')
        except Exception:
            return f"lr{v}"
    def year_short(y):
        return f"y{str(y)[-2:]}"
    model = hparams.get("model", "")
    model = "MCNN" if ("MultiChannelCNN" in model or old_model_name.startswith("MultiChannelCNN")) else (model[:8] or "MODEL")
    ds = hparams.get("downscaling")
    ds_part = f"d{ds}" if ds is not None else ""
    dropout = hparams.get("dropout", 0)
    do_part = f"do{int(round(float(dropout)*100))}"
    lr = lr_compact(hparams.get("learning_rate", 0))
    bs = hparams.get("batch_size")
    b_part = f"b{bs}" if bs is not None else ""
    ep = hparams.get("num_epochs") or hparams.get("epochs")
    e_part = f"e{ep}" if ep is not None else ""
    year = hparams.get("base_year") or hparams.get("year")
    y_part = year_short(year) if year else ""
    excluded = hparams.get("excluded_variables")
    ex_part = ""
    if excluded:
        ex_part = "ex_" + "+".join(excluded)
    finetune_year = None
    if hparams.get("finetuned_from") and hparams.get("year"):
        finetune_year = hparams.get("year")
    ft_part = f"ft{str(finetune_year)[-2:]}" if finetune_year else ""
    parts = [model, ds_part, do_part, lr, b_part, e_part, y_part, ex_part, ft_part]
    core = "_".join([p for p in parts if p])
    return core

def unique(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    parent = target.parent
    suffix = target.suffix
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def process(move: bool = False):
    if not ROOT.exists():
        print("saved_models introuvable")
        return
    json_files = list(ROOT.rglob("*.json"))
    if not json_files:
        print("Aucun JSON trouvé.")
        return
    print(f"{len(json_files)} fichiers JSON à traiter.")
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            old_name = data.get("model_name", jf.stem)
            hparams = extract_hparams(data)
            new_base = build_new_name(hparams, old_name)

            # Récupérer timestamp s'il apparaît dans l'ancien nom
            ts = None
            ts_match = re.search(r"\d{8}_\d{6}", jf.stem)
            if ts_match:
                ts = ts_match.group(0)
                new_base = f"{new_base}_{ts}"

            # Nouveau dossier
            new_dir = jf.parent.parent / new_base
            new_dir.mkdir(parents=True, exist_ok=True)

            # Mettre à jour JSON
            data["original_model_name"] = old_name
            data["model_name"] = new_base
            new_json = unique(new_dir / f"{new_base}.json")
            with open(new_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Poids .pth (copie du premier trouvé pertinent)
            pth_candidates = [p for p in jf.parent.glob("*.pth")]
            moved_weights = 0
            for pth in pth_candidates:
                # Stratégie simple: déplacer/copier tous les .pth du dossier
                new_pth = unique(new_dir / f"{new_base}.pth")
                (shutil.move if move else shutil.copy2)(pth, new_pth)
                moved_weights += 1
                break  # garder un seul (si plusieurs variantes inutile)

            # PNG (predictions / residuals / autres)
            png_candidates = list(jf.parent.glob("*.png"))
            moved_png = 0
            for png in png_candidates:
                # Identifier suffixe (predictions / residuals)
                suffix = ""
                if "pred" in png.stem.lower():
                    suffix = "_pred"
                elif "resid" in png.stem.lower():
                    suffix = "_resid"
                else:
                    suffix = ""
                new_png_name = f"{new_base}{suffix}.png"
                new_png = unique(new_dir / new_png_name)
                (shutil.move if move else shutil.copy2)(png, new_png)
                moved_png += 1

            # Optionnel: déplacer ou laisser l'ancien dossier
            if move:
                # Supprimer dossier source s'il est vide
                try:
                    for extra in jf.parent.iterdir():
                        # garder ce qui reste
                        pass
                    # (On ne supprime pas pour sécurité)
                except Exception:
                    pass

            print(f"[OK] {old_name} -> {new_base} (JSON, {moved_weights} pth, {moved_png} png)")
        except Exception as e:
            print(f"[SKIP] {jf}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compression des noms de fichiers (saved_models)")
    parser.add_argument("--move", action="store_true", help="Déplacer au lieu de copier (destructif)")
    args = parser.parse_args()
    process(move=args.move)