import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, WeightedRandomSampler)
from torchvision import datasets, transforms, models
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, balanced_accuracy_score, precision_recall_fscore_support)
from tqdm import tqdm
import time
import os
import warnings
warnings.filterwarnings('ignore')

#konstante
RANDOM_SEED    = 22
IMG_SIZE       = 224          # velicina ulazne slike
BATCH_SIZE     = 32
EPOCHS         = 30
LEARNING_RATE  = 0.0003
MIN_DELTA      = 0.0001
NUM_CLASSES    = 6
PATIENCE       = 7            # early stopping

#izvlacimo slike iz dataseta
DATA_DIR  = r'C:\Users\anaka\OneDrive\Radna površina\stidiranje vol8\VIM\D3\CNN_dataset_2026'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR  = os.path.join(DATA_DIR, 'test')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
#reproduktivnost rezultata
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CLASS_COLORS = ['#81eafc', '#ff6b6b', '#ffa500', '#a8ff78', '#bc92ff', '#fdec6d']

#stil def
def style_ax(ax):
    ax.set_facecolor('#17191d')
    ax.tick_params(colors='#aaaaaa')
    ax.grid(True, linestyle='--', alpha=0.1, color='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_color('#444444')

#stats
def print_stats(label, acc_train, acc_test, report):
    print(f"\n{'-'*50}")
    print(f"  {label}")
    print(f"{'-'*50}")
    print(f"  Tacnost (train) : {acc_train*100:.2f}%")
    print(f"  Tacnost (test)  : {acc_test*100:.2f}%")
    print(f"\n{report}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#augmentacija i transformacije
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#dataset
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=eval_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=eval_transforms)

NUM_CLASSES = len(train_dataset.classes)
CLASS_NAMES = train_dataset.classes

print(f"\nKlase ({NUM_CLASSES}): {CLASS_NAMES}")

#uravnotezenost klasa
class_counts = np.zeros(NUM_CLASSES, dtype=int)

for _, label in train_dataset.samples:
    class_counts[label] += 1

print("\nBroj slika po klasi u trening skupu:") 
for cls, cnt in zip(CLASS_NAMES, class_counts):
    print(f"{cls:15s}: {cnt}")

#weighted sampler
sample_weights = []
class_weights = 1.0 / class_counts

for _, label in train_dataset.samples:
    sample_weights.append(class_weights[label])

sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

#dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler,
                          num_workers=0, pin_memory=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

CLASS_NAMES = train_dataset.classes
print(f"\nKlase ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Trening  : {len(train_dataset)} slika")
print(f"Validacija: {len(valid_dataset)} slika")
print(f"Test     : {len(test_dataset)} slika")

#prikaz distribucije klasa
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#2f2f30')
style_ax(ax)

bars = ax.bar(CLASS_NAMES, class_counts, color=CLASS_COLORS, edgecolor='#17191d')
for bar, val in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

ax.set_xlabel('Klasa antene', color='#aaaaaa', fontsize=12)
ax.set_ylabel('Broj slika', color='#aaaaaa', fontsize=12)
ax.set_title('Distribucija klasa u trening skupu', color='white', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', colors='#aaaaaa')
print(os.getcwd())
plt.tight_layout()
plt.savefig('cnn_fig1_distribucija.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("\nSacuvano: cnn_fig1_distribucija.png")

#definisanje CNN modela
class SimpleCNN(nn.Module):
    """Manja custom CNN arhitektura"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            # Conv blok 1
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 112x112
            nn.Dropout2d(0.1),

            # Conv blok 2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 56x56
            nn.Dropout2d(0.1),

            # Conv blok 3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 28x28
            nn.Dropout2d(0.15),

            # Conv blok 4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 14x14
            nn.Dropout2d(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((2,2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128), nn.ReLU(), nn.Dropout(0.4), #probala 1*1
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

#transfer learing modeli

#resnet18
def make_resnet18(num_classes=6):
    model = models.resnet18(weights='IMAGENET1K_V1')

    #freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

#mobilenetv2
def make_mobilenet(num_classes=6):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    #freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

#evaluacija modela
def evaluate_model(model, loader, criterion=None):
    model.eval()

    total_loss=0
    correct=0
    total=0

    preds=[]
    labels=[]

    with torch.no_grad():
        for X,y in loader:
            X=X.to(DEVICE)
            y=y.to(DEVICE)
            out=model(X)

            if criterion is not None:
                loss=criterion(out, y)
                total_loss+=loss.item()*X.size(0)
            
            pred=out.argmax(1)

            correct+=(pred==y).sum().item()
            total+=X.size(0)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = correct / total

    if criterion is not None:
        total_loss /= total

    return total_loss, acc, np.array(preds), np.array(labels)


#funkcija za obucavanje
def train_model(model, train_loader, valid_loader, epochs=EPOCHS,
                lr=LEARNING_RATE, patience=PATIENCE, label='Model'):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4) #probala Adam obicni, mnogo sporo, nema L2 regularozaciju
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                      factor=0.5, patience=3) #ako tri epohe gubitak stagnira prepolovi lr rate
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_weights  = None
    no_improve    = 0

    print(f"\n{'='*55}")
    print(f"  Treniranje: {label}")
    print(f"{'='*55}")

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        #trening 
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * X.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total   += X.size(0)

        #validacija
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out  = model(X)
                loss = criterion(out, y)
                va_loss    += loss.item() * X.size(0)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total   += X.size(0)

        #racunamo prosek
        tr_loss /= tr_total
        va_loss /= va_total
        tr_acc   = tr_correct / tr_total
        va_acc   = va_correct / va_total

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        scheduler.step(va_loss) #prilagodjavamo lr na osnovu validacionog losa

        # early stopping
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()} #pamtimo najbolje tezine
            no_improve    = 0
            best_epoch=epoch
        else:
            no_improve += 1

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.1f}% | "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:.1f}%"
              + (" [*]" if no_improve == 0 else ""))

        if no_improve >= patience:
            print(f"\n  Early stopping u epohi {epoch}.")
            break

    elapsed = time.time() - t_start
    print(f"\n  Ukupno vreme treniranja: {elapsed:.1f} s")

    if best_weights is not None:
        model.load_state_dict(best_weights) #vraca model koji je bio najbolji na validaciji a ne onaj gde su stali ako early stop
    return model, history, best_epoch


#takmicenje izmedju arhitektura


architectures = {
    'SimpleCNN'     : SimpleCNN(NUM_CLASSES),
    'ResNet18'      : make_resnet18(NUM_CLASSES),
    'MobileNetV2'   : make_mobilenet(NUM_CLASSES),
}

arch_results  = {}
arch_histories = {}

print("\nPoredjenje arhitektura CNN mreze...")
for arch_name, arch_model in architectures.items():
    model_trained, history, best_epoch = train_model(
        arch_model, train_loader, valid_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
        label=arch_name
    )
    val_acc = history['val_acc'][best_epoch - 1] #uzimamo vrednost iz istorije za najbolju epohu

    arch_results[arch_name] = {
        'acc': val_acc,
        'model': model_trained,
        'history': history,
        'epoch': best_epoch
    }
    arch_histories[arch_name] = history
    print(f"\n  {arch_name}: val_acc = {val_acc*100:.2f}%")
#izvlacimo najbolju arhitekturu
best_arch_name = max(arch_results, key=lambda k: arch_results[k]['acc'])
best_model     = arch_results[best_arch_name]['model']
best_history   = arch_results[best_arch_name]['history']
best_epoch     = arch_results[best_arch_name]['epoch']
print(f"\n  Optimalna arhitektura: {best_arch_name} "
      f"(val_acc = {arch_results[best_arch_name]['acc']*100:.2f}%)")

#grafik poredjenja arhitektura
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#2f2f30')
style_ax(ax)

vals   = [arch_results[k]['acc'] * 100 for k in arch_results]
labels = list(arch_results.keys())
colors = ['#ff6b6b' if k == best_arch_name else '#81eafc' for k in labels]

bars = ax.bar(labels, vals, color=colors, edgecolor='#17191d')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
            f'{val:.1f}%', ha='center', va='bottom', color='white',
            fontsize=11, fontweight='bold')

ax.set_xlabel('Arhitektura', color='#aaaaaa', fontsize=12)
ax.set_ylabel('Tacnost na validaciji [%]', color='#aaaaaa', fontsize=12)
ax.set_title('Poredjenje CNN arhitektura', color='white', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(handles=[mpatches.Patch(color='#ff6b6b', label=f'Optimalna: {best_arch_name}')],
          facecolor='#17191d', labelcolor='white', loc='lower right')

plt.tight_layout()
plt.savefig('cnn_fig2_arhitekture.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("\nSacuvano: cnn_fig2_arhitekture.png")


# grafik procesa obucavanja (za finalnu arhitekturu)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#2f2f30')
for ax in axes:
    style_ax(ax)

ep = range(1, len(best_history['train_loss']) + 1)
#gubitak/loss
axes[0].plot(ep, best_history['train_loss'], color='#81eafc', linewidth=2,
             label='Trening gubitak')
axes[0].plot(ep, best_history['val_loss'], color='#ff6b6b', linewidth=2,
             linestyle='--', label='Validacioni gubitak')
axes[0].axvline(best_epoch, linestyle=':', alpha=0.8, linewidth=2, color='#bc92ff', 
                label=f'Optimalna epoha ({best_epoch})')
axes[0].set_xlabel('Epoha', color='#aaaaaa', fontsize=11)
axes[0].set_ylabel('Gubitak (Cross-Entropy)', color='#aaaaaa', fontsize=11)
axes[0].set_title(f'Gubitak kroz epohe - {best_arch_name}', color='white',
                  fontsize=13, fontweight='bold')
axes[0].legend(facecolor='#17191d', labelcolor='white')
#tacnost/accuracy
train_acc_plot = [v * 100 for v in best_history['train_acc']]
val_acc_plot   = [v * 100 for v in best_history['val_acc']]
axes[1].plot(ep, train_acc_plot, color='#81eafc', linewidth=2.5, label='Trening tacnost')
axes[1].plot(ep, val_acc_plot, color='#ff6b6b', linewidth=2.5, linestyle='--', label='Validaciona tacnost')

#najmanji loss
best_acc_val = val_acc_plot[best_epoch-1]
axes[1].scatter(best_epoch, best_acc_val, color='#bc92ff', s=100, zorder=5, 
                edgecolors='white', label=f'Optimalna tačnost ({best_acc_val:.2f}%)')
axes[1].set_xlabel('Epoha', color='#aaaaaa', fontsize=11)
axes[1].set_ylabel('Tacnost [%]', color='#aaaaaa', fontsize=11)
axes[1].set_title(f'Tacnost kroz epohe -  {best_arch_name}', color='white',
                  fontsize=13, fontweight='bold')
axes[1].legend(facecolor='#17191d', labelcolor='white')

plt.tight_layout()
plt.savefig('cnn_fig3_obucavanje.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Sacuvano: cnn_fig3_obucavanje.png")

# evaluacija finalnog modela na TEST skupu

print(f"\nEvaluacija na test skupu ({best_arch_name})...")
best_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE)
        out = best_model(X)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

#tacnost na test skupu
test_acc = np.mean(all_preds == all_labels)
#tacnost na trening skupu iz optimalne epohe
train_acc = best_history['train_acc'][best_epoch-1]


report = classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=4)
print_stats(f"Finalni model — {best_arch_name}", train_acc, test_acc, report)

#matrica konfuzije

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#17191d')

im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
cb = plt.colorbar(im, ax=ax)
cb.ax.yaxis.set_tick_params(color='white', labelcolor='white')
cb.outline.set_visible(False)

ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', color='#aaaaaa')
ax.set_yticklabels(CLASS_NAMES, color='#aaaaaa')

thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f'{cm[i, j]}',
                ha='center', va='center',
                color='white' if cm[i, j] < thresh else 'black',
                fontsize=12, fontweight='bold')

ax.set_xlabel('Prediktovana klasa', color='#aaaaaa', fontsize=12)
ax.set_ylabel('Stvarna klasa', color='#aaaaaa', fontsize=12)
ax.set_title(f'Matrica konfuzije — {best_arch_name} (test skup)',
             color='white', fontsize=14, fontweight='bold')
ax.grid(False)

plt.tight_layout()
plt.savefig('cnn_fig4_konfuzija.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Sacuvano: cnn_fig4_konfuzija.png")


#prikaz filtara iz prvog konvolucionog sloja

first_conv = next((m for m in best_model.modules() if isinstance(m, nn.Conv2d)), None)
if first_conv is not None:
    weights = first_conv.weight.data.cpu()
    n_filters = min(32, weights.shape[0])

    # normalizacija na [0,1] za prikaz
    w_min, w_max = weights.min(), weights.max()
    weights_norm = (weights - w_min) / (w_max - w_min + 1e-8)

    cols = 8
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), squeeze=False)
    fig.patch.set_facecolor('#1a1a2e')

    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        if idx < n_filters:
            # prikazujemo RGB ili prvih 3 kanala
            filt = weights_norm[idx].permute(1, 2, 0).numpy()
            if filt.shape[2] == 1:
                ax.imshow(filt[:, :, 0], cmap='viridis', interpolation='nearest')
            else:
                filt = np.clip(filt[:, :, :3], 0, 1)
                ax.imshow(filt, interpolation='nearest')
        ax.axis('off')
        ax.set_facecolor('#17191d')

    plt.suptitle(f'Filtri iz prvog konvolucionog sloja — {best_arch_name}',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('cnn_fig5_filtri.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("Sacuvano: cnn_fig5_filtri.png")


# mapa obelezja (feature map) za prvi konvolucion sloj

# uzimamo prvu sliku iz test dataseta
sample_img, sample_label = test_dataset[0]
sample_img_batch = sample_img.unsqueeze(0).to(DEVICE)

activation = {}
def hook_fn(module, inp, out):
    activation['fmap'] = out.detach().cpu()

hook = first_conv.register_forward_hook(hook_fn)
best_model.eval()
with torch.no_grad():
    _ = best_model(sample_img_batch)
hook.remove()

fmaps = activation['fmap'][0]          # [C, H, W]

n_features = min(8, fmaps.shape[0]) 
cols_fmap = 4
rows_fmap = (n_features + cols_fmap - 1) // cols_fmap                         

fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor('#1a1a2e')
ax_orig = plt.subplot2grid((rows_fmap, 2 * cols_fmap), (0, 0), rowspan=rows_fmap, colspan=cols_fmap)
ax_orig.set_facecolor('#17191d')
ax_orig.axis('off')

# originalna slika (denormalizacija)
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
img_show = sample_img.permute(1, 2, 0).numpy() * std + mean
img_show = np.clip(img_show, 0, 1)
ax_orig.imshow(img_show)
ax_orig.set_title(f'Ulazna slika\n(Klasa: {CLASS_NAMES[sample_label]})',
                  color='white', fontsize=12, fontweight='bold', pad=10)

#podgrafici za mape obelezja, desno
for idx in range(n_features):
    r = idx // cols_fmap
    c = idx % cols_fmap + cols_fmap 
    
    ax_fmap = plt.subplot2grid((rows_fmap, 2 * cols_fmap), (r, c))
    ax_fmap.set_facecolor('#17191d')
    ax_fmap.axis('off')

    ax_fmap.imshow(fmaps[idx].numpy(), cmap='inferno')
    ax_fmap.set_title(f'Filtar {idx}', color='#aaaaaa', fontsize=10)

plt.suptitle(f'Prikaz aktivacija (mape obeležja prvog sloja) — {best_arch_name}', 
             color='white', fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('cnn_fig6_aktivacija.png', dpi=150, bbox_inches='tight', facecolor='#121212')
plt.close()

print("Sacuvano: cnn_fig6_aktivacija.png")

#ispravno klasifikovano slike

def occlusion_sensitivity(model, img_tensor, true_label, patch_size=32, stride=16):
    model.eval()
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    sens_map = np.zeros((H, W))
    entry_count = np.zeros((H, W))

    #bazna verovatnoca
    with torch.no_grad():
        base_prob = torch.softmax(model(img_tensor.unsqueeze(0).to(DEVICE)), dim=1)
        base_p    = base_prob[0, true_label].item()

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = img_tensor.clone()
            #sivi/nulti prozor (okluzija)
            occluded[:, y:y+patch_size, x:x+patch_size] = 0.0
            with torch.no_grad():
                prob = torch.softmax(
                    model(occluded.unsqueeze(0).to(DEVICE)), dim=1)
                p = prob[0, true_label].item()

            sens_map[y:y+patch_size, x:x+patch_size] += (base_p - p)
            entry_count[y:y+patch_size, x:x+patch_size] += 1
    
    sens_map = sens_map / (entry_count + 1e-8) #preklapaju nam se prozori
    # normalizacija na [o,1]
    sens_map = (sens_map - sens_map.min()) / (sens_map.max() - sens_map.min() + 1e-8)
    return sens_map

# pronalazimo ispravno i pogresno klasifikovanu sliku iz test skupa
correct_idx   = None
incorrect_idx = None

best_model.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        img_t, lbl = test_dataset[i]
        out   = best_model(img_t.unsqueeze(0).to(DEVICE))
        pred  = out.argmax(1).item()
        if correct_idx is None and pred == lbl:
            correct_idx = (i, img_t, lbl, pred)
        if incorrect_idx is None and pred != lbl:
            incorrect_idx = (i, img_t, lbl, pred)
        if correct_idx and incorrect_idx:
            break

# occlusion sensitivity za ispravno klasifikovanu sliku
i_c, img_c, lbl_c, pred_c = correct_idx

print(f"\nRacunam occlusion sensitivity mapu za {best_arch_name}")
t0 = time.time()
sens = occlusion_sensitivity(best_model, img_c, lbl_c)
print(f"  Gotovo za {time.time()-t0:.1f} s")

# top 3 klase za ispravnu sliku
best_model.eval()
with torch.no_grad():
    probs = torch.softmax(best_model(img_c.unsqueeze(0).to(DEVICE)), dim=1)[0]
top3_idx = probs.argsort(descending=True)[:3].cpu().numpy()
top3     = [(CLASS_NAMES[j], probs[j].item() * 100) for j in top3_idx]

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

img_c_show = img_c.permute(1, 2, 0).numpy() * std + mean
img_c_show = np.clip(img_c_show, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#17191d')
#ulazna originalna slika
axes[0].imshow(img_c_show)
axes[0].set_title(f'Ispravno klasifikovana slika\nStvarno: {CLASS_NAMES[lbl_c]} | '
                  f'Prediktovano: {CLASS_NAMES[pred_c]}',
                  color='white', fontsize=11, fontweight='bold')
axes[0].axis('off')

# occlusion mapa preklopljena preko slike
axes[1].imshow(img_c_show)
im_oc = axes[1].imshow(sens, cmap='jet', alpha=0.5,
                        vmin=0, vmax=1, interpolation='bilinear')
plt.colorbar(im_oc, ax=axes[1]).ax.yaxis.set_tick_params(color='white', labelcolor='white')
top3_str = ' | '.join([f'{cls}: {p:.1f}%' for cls, p in top3])
axes[1].set_title(f'Occlusion Sensitivity mapa (Alpha Data=0.5)\nTop-3: {top3_str}',
                  color='white', fontsize=10, fontweight='bold')
axes[1].axis('off')

plt.suptitle('Analiza ispravno klasifikovane slike', color='white',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cnn_fig7_occlusion.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Sacuvano: cnn_fig7_occlusion.png")

print(f"\nTop-3 klase za ispravnu sliku ({CLASS_NAMES[lbl_c]}):")
for cls, p in top3:
    print(f"  {cls:12s}: {p:.2f}%")

#pogresno klasifikovana slika

i_w, img_w, lbl_w, pred_w = incorrect_idx
img_w_show = img_w.permute(1, 2, 0).numpy() * std + mean
img_w_show = np.clip(img_w_show, 0, 1)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#17191d')
ax.imshow(img_w_show)
ax.set_title(f'Pogresno klasifikovana slika\n'
             f'Stvarno: {CLASS_NAMES[lbl_w]} | Prediktovano: {CLASS_NAMES[pred_w]}',
             color='white', fontsize=12, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('cnn_fig8_pogresna.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Sacuvano: cnn_fig8_pogresna.png")


#rezime SVEGA

print(f"""
{'='*60}
  Finalni rezime — CNN klasifikacija antena
{'='*60}

  Optimalna arhitektura : {best_arch_name}
  Optimizer             : Adamw | Adaptive LR (ReduceLROnPlateau)
  Mini-batch size       : {BATCH_SIZE}
  Learning rate         : {LEARNING_RATE}
  Augmentacija          : HorizontalFlip, Rotation(15), ColorJitter
  Normalizacija         : ImageNet mean/std

  Tacnost na uzorkovanim trening slikama     : {train_acc*100:.2f}%
  Tacnost na test skupu                      : {test_acc*100:.2f}%

  Poredjenje arhitektura:
""")
for k, v_dict in arch_results.items():
    marker = ' <-- optimalna' if k == best_arch_name else ''
    print(f"    {k:15s}: {v_dict['acc']*100:.2f}%{marker}")


