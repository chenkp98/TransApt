import torch
import torch.nn as nn
import pandas as pd
from model import TransApt, GVP

NUCLEOTIDES = ['X', 'A', 'T', 'C', 'G']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

def vector_to_scalar(seq_path):

    data = pd.read_csv(seq_path)
    max_length = max(len(seq) for seq in data['Sequence'])
    padded_sequences = [seq + 'X' * (max_length - len(seq)) for seq in data['Sequence']]

    data_change = torch.empty(len(padded_sequences), max_length)
    for j in range(len(padded_sequences)):
        seq = padded_sequences[j]

        for i in range(100):
            if seq[i] == 'A':
                data_change[j][i]=1
            elif seq[i] == 'T':
                data_change[j][i]=2
            elif seq[i] == 'C':
                data_change[j][i]=3
            elif seq[i] == 'G':
                data_change[j][i]=4
            elif seq[i] == 'X':
                data_change[j][i]=0

    return data_change

def calculate_gc_content(seq):
    if len(seq) == 0: return 0
    return (seq.count('G') + seq.count('C')) / len(seq)

if __name__ == "__main__":
    seq_path = './input/zhilian_seq.csv'
    targets = vector_to_scalar(seq_path)
    scalar_features = torch.load('./input/zhilian_diherals.pt')
    coords = torch.load('./input/zhilian_coords.pt').reshape(6490, 100, -1)
    
    dataset = TensorDataset(coords, scalar_features, targets)
    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_val_dataset = Subset(dataset, train_val_idx)
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (cv_train_idx, cv_val_idx) in enumerate(kfold.split(train_val_dataset)):
        print(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")
        
        fold_train_sub = Subset(train_val_dataset, cv_train_idx)
        fold_val_sub = Subset(train_val_dataset, cv_val_idx)
        train_loader = DataLoader(fold_train_sub, batch_size=16, shuffle=True)
        val_loader = DataLoader(fold_val_sub, batch_size=16, shuffle=False)
        
        model = TransApt().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(50):
            model.train()
            for b_coords, b_scalar, b_targets in train_loader:
                b_coords, b_scalar = b_coords.to(device), b_scalar.to(device)
                b_targets = b_targets.to(device).long()
                optimizer.zero_grad()
                output = model(b_coords, b_scalar)
                loss = criterion(output.view(-1, 5), b_targets.view(-1))
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for v_coords, v_scalar, v_targets in val_loader:
                    v_coords, v_scalar = v_coords.to(device), v_scalar.to(device)
                    v_targets = v_targets.to(device).long()
                    v_out = model(v_coords, v_scalar)
                    val_losses.append(criterion(v_out.view(-1, 5), v_targets.view(-1)).item())
            
            avg_val_loss = np.mean(val_losses)
            print(f"Fold {fold+1} | Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | PPL: {np.exp(avg_val_loss):.2f}")

        torch.save(model.state_dict(), os.path.join(output_dir, f'model_fold_{fold+1}.pth'))

        print(f"Evaluating independent test set using Fold {fold+1} model...")
        fold_test_results = []
        fold_true_bases = []
        fold_pred_bases = []

        model.eval()
        with torch.no_grad():
            for i, (t_coords, t_scalar, t_targets) in enumerate(test_loader):
                real_len = (t_targets != 0).sum().item()
                orig_indices = t_targets[0, :real_len].long().tolist()
                original_seq = ''.join([NUCLEOTIDES[idx] for idx in orig_indices])
                orig_gc = calculate_gc_content(original_seq)
                
                input_coords = t_coords.to(device)
                input_scalar = t_scalar.to(device)
                logits = model(input_coords, input_scalar)
                
                valid_logits = logits[0, :real_len, :]
                probs = F.softmax(valid_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                probs = torch.clamp(probs, min=1e-8)
                
                samples = torch.multinomial(probs, num_samples=1)
                gen_indices = samples.view(-1).long().tolist()
                gen_seq = ''.join([NUCLEOTIDES[idx] for idx in gen_indices])
                
                matches = sum(1 for a, b in zip(original_seq, gen_seq) if a == b)
                recovery = matches / real_len
                gen_gc = calculate_gc_content(gen_seq)
                
                fold_test_results.append({
                    "Fold": fold + 1,
                    "Struct_ID": test_idx[i],
                    "Recovery": recovery,
                    "GC_Error": abs(gen_gc - orig_gc),
                    "Gen_Seq": gen_seq
                })
                
                fold_true_bases.extend(orig_indices)
                fold_pred_bases.extend(gen_indices)

        df_fold_test = pd.DataFrame(fold_test_results)
        df_fold_test.to_csv(os.path.join(output_dir, f'test_eval_fold_{fold+1}.csv'), index=False)
        
        print(f">> Fold {fold+1} 测试集平均恢复率: {df_fold_test['Recovery'].mean():.2%}")
        
        cm = confusion_matrix(fold_true_bases, fold_pred_bases, labels=[1, 2, 3, 4])
        print(f">> Fold {fold+1} 混淆矩阵:\n{cm}")

    print("\nTraining and evaluation for all folds have been completed.")