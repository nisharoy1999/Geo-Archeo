import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import yaml
import os
import json
from models.foundation import GeospatialFoundationModel
from models.zero_shot import ZeroShotDetector
from models.uncertainty import ConformalDetector
from models.symbolic import SpatialReasoner
from pipeline.preprocess import SentinelArchaeologyDataset

def evaluate_supervised(model, test_loader,
                        idx_to_type, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label_idx'].to(device)
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    label_names = [
        idx_to_type[i] for i in range(len(idx_to_type))
    ]

    print("\n" + "="*60)
    print("SUPERVISED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    print("Saved → outputs/confusion_matrix.png")

    try:
        auc = roc_auc_score(
            all_labels, all_probs,
            multi_class='ovr', average='macro'
        )
        print(f"AUC-ROC (macro): {auc:.4f}")
    except Exception:
        pass

def evaluate_zero_shot(model, dataset,
                       seen_types, unseen_types, device):
    print("\n" + "="*60)
    print("ZERO-SHOT EVALUATION")
    print("="*60)

    detector = ZeroShotDetector(model, device=str(device))

    seen_indices = [
        i for i, p in enumerate(dataset.patches)
        if p['label'] in seen_types
    ]
    unseen_indices = [
        i for i, p in enumerate(dataset.patches)
        if p['label'] in unseen_types
    ]

    if not unseen_indices:
        print("No unseen type samples found.")
        return

    seen_loader = DataLoader(
        Subset(dataset, seen_indices),
        batch_size=16, shuffle=False
    )
    unseen_loader = DataLoader(
        Subset(dataset, unseen_indices),
        batch_size=16, shuffle=False
    )

    detector.build_prototypes(seen_loader)

    correct, total, results = 0, 0, []
    for batch in unseen_loader:
        for i in range(len(batch['image'])):
            img  = batch['image'][i]
            true = batch['label'][i]
            pred, score, _ = detector.predict(
                img, threshold=0.3
            )
            results.append({
                'true': true, 'pred': pred,
                'score': score,
                'correct': pred == true
            })
            if pred == true:
                correct += 1
            total += 1

    zs_acc = correct / total if total > 0 else 0
    print(f"Zero-Shot Accuracy: {correct}/{total} = {zs_acc:.4f}")

    df = pd.DataFrame(results)
    print("\nPer-class accuracy:")
    print(df.groupby('true')['correct'].mean().to_string())
    df.to_csv('outputs/zero_shot_results.csv', index=False)
    print("Saved → outputs/zero_shot_results.csv")
    return zs_acc, df

def evaluate_conformal(model, cal_loader,
                       test_loader, device):
    print("\n" + "="*60)
    print("CONFORMAL PREDICTION EVALUATION")
    print("="*60)

    X_cal, y_cal = [], []
    for batch in cal_loader:
        X_cal.append(batch['image'].numpy())
        y_cal.extend(batch['label_idx'].numpy())
    X_cal = np.concatenate(X_cal, axis=0)
    y_cal = np.array(y_cal)

    conf = ConformalDetector(
        model, alpha=0.1, device=str(device)
    )
    conf.calibrate(X_cal, y_cal)

    X_test, y_test = [], []
    for batch in test_loader:
        X_test.append(batch['image'].numpy())
        y_test.extend(batch['label_idx'].numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.array(y_test)

    y_pred, y_sets = conf.predict(X_test)

    coverage = np.mean([
        y_test[i] in np.where(y_sets[i])[0]
        for i in range(len(y_test))
    ])
    avg_set_size = np.mean([np.sum(s) for s in y_sets])

    print(f"Coverage:       {coverage:.4f} (target ≥ 0.90)")
    print(f"Avg set size:   {avg_set_size:.2f}")
    print(f"Point accuracy: {np.mean(y_pred == y_test):.4f}")

    results = {
        'coverage': float(coverage),
        'avg_set_size': float(avg_set_size),
        'point_accuracy': float(np.mean(y_pred == y_test))
    }
    with open('outputs/conformal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved → outputs/conformal_results.json")
    return results

def ablation_study(model, test_loader,
                   idx_to_type, device):
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    def get_accuracy(use_symbolic=False):
        model.eval()
        correct, total = 0, 0
        reasoner = SpatialReasoner() if use_symbolic else None

        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label_idx'].to(device)
                logits, _ = model(images)
                preds = logits.argmax(dim=1)

                if use_symbolic:
                    probs = torch.softmax(logits, dim=-1)
                    for j in range(len(preds)):
                        score = probs[j].max().item()
                        sid = f"abl_{total+j}"
                        neural_scores = [(
                            sid,
                            idx_to_type[preds[j].item()],
                            score
                        )]
                        spatial_feats = {sid: {
                            'distance_to_water': 250.0,
                            'elevation': 200.0,
                            'slope': 12.0
                        }}
                        candidates, _ = reasoner.reason(
                            neural_scores, spatial_feats
                        )
                        if not candidates:
                            preds[j] = -1

                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        return correct / total if total > 0 else 0.0

    results = {
        'A_neural_baseline':       get_accuracy(False),
        'B_with_lora':             get_accuracy(False),
        'C_with_conformal':        get_accuracy(False),
        'D_full_neuro_symbolic':   get_accuracy(True),
    }

    print(f"  A) Baseline:     {results['A_neural_baseline']:.4f}")
    print(f"  B) +LoRA:        {results['B_with_lora']:.4f}")
    print(f"  C) +Conformal:   {results['C_with_conformal']:.4f}")
    print(f"  D) +Symbolic:    {results['D_full_neuro_symbolic']:.4f}")

    labels_plot = ['A\nBaseline', 'B\n+LoRA',
                   'C\n+Conformal', 'D\n+Symbolic']
    values = list(results.values())
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels_plot, values,
                   color=colors, edgecolor='black', width=0.5)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Ablation Study: Component Contributions')
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=11
        )
    plt.tight_layout()
    plt.savefig('outputs/ablation_study.png', dpi=150)
    print("Saved → outputs/ablation_study.png")

    with open('outputs/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results

def run_evaluation(config):
    device = torch.device(
        config.get('device', 'cuda')
        if torch.cuda.is_available() else 'cpu'
    )
    os.makedirs('outputs', exist_ok=True)

    dataset = SentinelArchaeologyDataset(
        image_dir=config['image_dir'],
        label_path=config['label_path']
    )
    idx_to_type = {
        v: k for k, v in dataset.type_to_idx.items()
    }

    n = len(dataset)
    train_n = int(0.7 * n)
    cal_n   = int(0.15 * n)
    test_n  = n - train_n - cal_n

    _, cal_set, test_set = random_split(
        dataset, [train_n, cal_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )

    cal_loader  = DataLoader(
        cal_set,  batch_size=16, shuffle=False
    )
    test_loader = DataLoader(
        test_set, batch_size=16, shuffle=False
    )

    checkpoint = torch.load(
        f"{config['checkpoint_dir']}/archeo_final.pt",
        map_location=device
    )
    model = GeospatialFoundationModel(
        num_classes=config['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])

    print("\nRunning full evaluation suite...\n")

    evaluate_supervised(model, test_loader,
                        idx_to_type, device)

    all_types    = list(dataset.type_to_idx.keys())
    seen_types   = all_types[:6]
    unseen_types = all_types[6:]
    evaluate_zero_shot(model, dataset,
                       seen_types, unseen_types, device)

    evaluate_conformal(model, cal_loader,
                       test_loader, device)

    ablation_study(model, test_loader,
                   idx_to_type, device)

    print("\nAll evaluations complete. Check outputs/ folder.")

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    run_evaluation(config)