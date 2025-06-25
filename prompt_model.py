import argparse
import random,os
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader, Dataset
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from load import *
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, \
    classification_report, balanced_accuracy_score
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")
####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--task_name', type=str, default='deepcdr', help='task name')
parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
parser.add_argument('--output_type', type=str, default='cell',choices=['cell','gene','gene_batch','gene_expression'], help='cell or gene embedding; default: cell the difference between gene and gene_batch is that in gene mode the gene embedding will be processed one by one. while in gene_batch mode, the gene embedding will be processed in batch. GEARS use gene_batch mode.')
parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
parser.add_argument('--tgthighres', type=str, default='t4', help='the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion. only valid for input_type=singlecell')
parser.add_argument('--data_path', type=str, default='./data/ms/0/ms_test0.h5ad', help='input data path')
parser.add_argument('--save_path', type=str, default='./', help='save path')
parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
parser.add_argument('--demo', action='store_true', default=False, help='if demo, only infer 10 samples')
parser.add_argument('--version',  type=str, default='ce', help='only valid for output_type=cell. For read depth enhancemnet, version=rde For others, version=ce')
parser.add_argument('--model_path',  type=str, default='None', help='pre-trained model path')
parser.add_argument('--ckpt_name',  type=str, default='01B-resolution', help='checkpoint name')
parser.add_argument("--peft_type", type=str, default='Encoder_adapter',help=' Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune')
parser.add_argument("--use_prompt", type=bool, default=True)
parser.add_argument("--fold_idx", type=str, default='0')
parser.add_argument("--data_name", type=str, default='ms',help='NSCLC/COVID/ms')
parser.add_argument("--batch_size", type=int, default=32, help='Number of batch size.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--learning_rate", type=float, default=5e-4, help='Learning rate.')

args = parser.parse_args()

n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # token
mlp_adapter_conf=[1, 1, 1, 1, 1, 1, 0,0,0,0,0,0]
space_adapter_conf=[1, 1, 1, 1, 1, 1,0,0,0,0,0,0]
# mlp_adapter_conf=[0,0,0,0,0,0, 1, 1, 1, 1, 1, 1]
# space_adapter_conf=[0,0,0,0,0,0, 1, 1, 1, 1, 1, 1]
peft_prompt_relationship = {
    "Encoder_adapter": "encoder-prompt",
    "Token_adapter": "head-prompt",
    "Prefix": "prefix-prompt",
    "LoRA": "LoRA",
    "finetune": "finetune"
}

prompt_type = peft_prompt_relationship[args.peft_type]
prompt_settings = {
    "use_prompt": args.use_prompt,
    "num_tokens": 64,
    "prompt_type": prompt_type,
    "n_layers_conf": n_layers_conf,
    "mlp_adapter_conf": mlp_adapter_conf,
    "space_adapter_conf": space_adapter_conf
}

ckpt_dir = args.save_path
LEARNING_RATE = args.learning_rate
####################################Settings#################################

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    print("mapping gene num:" + str(len(gene_list) - len(to_fill_columns)))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                              columns=to_fill_columns,
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                        index=X_df.index,
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var


# def load_data(args):
#
#     gexpr_feature = sc.read_h5ad(args.data_path)
#     idx = gexpr_feature.obs_names.tolist()
#     label = gexpr_feature.obs["celltype"]
#     try:
#         col = gexpr_feature.var.gene_name.tolist()
#     except:
#         col = gexpr_feature.var_names.tolist()
#     if issparse(gexpr_feature.X):
#         gexpr_feature = gexpr_feature.X.toarray()
#     else:
#         gexpr_feature = gexpr_feature.X
#     gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
#
#     if gexpr_feature.shape[1] < 19264:
#         print('covert gene feature into 19264')
#         gexpr_feature, to_fill_columns, var = main_gene_selection(gexpr_feature, gene_list)
#         assert gexpr_feature.shape[1] >= 19264
#
#     if args.pre_normalized == 'F':
#         adata = sc.AnnData(gexpr_feature)
#         sc.pp.normalize_total(adata)
#         sc.pp.log1p(adata)
#         gexpr_feature = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
#
#     print(gexpr_feature.shape)
#     return gexpr_feature, label


class SCDataset(Dataset):
    def __init__(self, adata, gene_list=None,label_to_int=None, transform=None):
        """
        Args:
            args: object with attributes:
                - data_path: path to the .h5ad file
                - pre_normalized: 'T' or 'F', whether the data is already normalized
            gene_list: full list of target genes (length >= 19264)
            transform: optional transform to be applied on a sample
        """
        self.transform = transform
        self.gene_list = gene_list
        self.label_to_int = label_to_int
        # Load and process data
        # adata = sc.read_h5ad(data_path)
        idx = adata.obs_names.tolist()
        # _, label = np.unique(np.array(adata.obs["celltype"]), return_inverse=True)
        label = adata.obs[celltype_key].map(label_to_int).to_numpy()

        try:
            col = adata.var.gene_name.tolist()
        except:
            col = adata.var_names.tolist()

        if issparse(adata.X):
            gexpr = adata.X.toarray()
        else:
            gexpr = adata.X

        gexpr = pd.DataFrame(gexpr, index=idx, columns=col)

        if gexpr.shape[1] < 19264:
            assert self.gene_list is not None, "gene_list must be provided when gene count < 19264"
            gexpr, _, _ = main_gene_selection(gexpr, self.gene_list)
            assert gexpr.shape[1] >= 19264

        if args.pre_normalized == 'F':
            print("preprocess data")
            adata = sc.AnnData(gexpr)
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            gexpr = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

        self.features = torch.tensor(gexpr.values, dtype=torch.float32)
        self.label = label  # categorical string, optionally can map to int

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        expression = self.features[idx]
        label = self.label[idx]
        if self.transform:
            expression = self.transform(expression)
        return expression, label



class LinearProbingClassifier(nn.Module):

    def __init__(self, ckpt_path,prompt_settings,key,n_class,pool_type,frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore
        self.key = key
        self.prompt_settings = prompt_settings
        self.n_class = n_class
        self.pool_type = pool_type

    def build(self):
        model,model_config = load_model_frommmf(self.ckpt_path, self.prompt_settings, self.key)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        if self.pool_type == 'all':
            self.fc1 = nn.Sequential(
                nn.Linear(model_config['encoder']['hidden_dim']*4, 256),
                nn.ReLU(),
                nn.Linear(256, self.n_class)  # ['n_class']
            )
        elif self.pool_type == 'max':
            self.fc1 = nn.Sequential(
                nn.Linear(model_config['encoder']['hidden_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, self.n_class)  # ['n_class']
            )
        if self.pool_type == 'all':
            # self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim']*4, affine=True, eps=1e-6)
            self.norm = torch.nn.LayerNorm(model_config['encoder']['hidden_dim'] * 4)
        elif self.pool_type == 'max':
            # self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=True, eps=1e-6)
            self.norm = torch.nn.LayerNorm(model_config['encoder']['hidden_dim'])

        self.model_config = model_config

        # if self.frozenmore:
        #     for _,p in self.token_emb.named_parameters():
        #         p.requires_grad = False
        #     for _,p in self.pos_emb.named_parameters():
        #         p.requires_grad = False
        #     print('self.pos_emb and self.token_emb also frozen')
        #
        # for na, param in self.encoder.named_parameters():
        #     param.requires_grad = False
        # for na, param in self.encoder.transformer_encoder[-2].named_parameters():
        #     print('self.encoder.transformer_encoder ',na,' have grad')
        #     param.requires_grad = True

        # pre_freeze_param_count = sum(
        #     dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
        keywords = ('lora', 'adapter', 'Adapter','prompt_embeddings')
        for name, para in model.named_parameters():
            para.requires_grad = False
        params_to_update = filter(lambda p: any(keyword in p[0] for keyword in keywords),
                                  model.named_parameters())
        for _, param in params_to_update:
            param.requires_grad = True
        for na, param in self.fc1.named_parameters():
            param.requires_grad = True
        for na, param in self.norm.named_parameters():
            param.requires_grad = True


        
    def forward(self, data, *args, **kwargs):

        x = data # (B, L)
        # print(x.shape)
        value_labels = x > 0

        # if args.tgthighres[0] == 'f':
        #     pretrain_gene_x = torch.tensor(
        #         tmpdata + [np.log10(totalcount * float(args.tgthighres[1:])), np.log10(totalcount)]).unsqueeze(0).cuda()
        # elif args.tgthighres[0] == 'a':
        #     pretrain_gene_x = torch.tensor(
        #         tmpdata + [np.log10(totalcount) + float(args.tgthighres[1:]), np.log10(totalcount)]).unsqueeze(0).cuda()
        # elif args.tgthighres[0] == 't':
        #     pretrain_gene_x = torch.tensor(tmpdata + [float(args.tgthighres[1:]), np.log10(totalcount)]).unsqueeze(
        #         0).cuda()
        # else:
        #     raise ValueError('tgthighres must be start with f, a or t')

        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        # print(x.shape, flush=True)
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x,x_padding)
        # print(logits.shape, flush=True)
        # print("after encoder")
        # print(logits[0:20])
        # mlp
        geneemb1 = logits[:, -1, :]
        geneemb2 = logits[:, -2, :]
        geneemb3, _ = torch.max(logits[:, :-2, :], dim=1)
        geneemb4 = torch.mean(logits[:,:-2,:], dim=1)
        if self.pool_type == 'all':
            geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
        elif self.pool_type == 'max':
            geneembmerge, _ = torch.max(logits, dim=1)

        # mlp
        # logits, _ = torch.max(logits, dim=1)  # b,dim
        # print(logits.shape)
        # print("after max pooling")
        # print(geneembmerge[0:20])
        logits = self.norm(geneembmerge)
        # print("after norm")
        # print(logits[0:20])
        logits = self.fc1(logits)
        # print("logits")
        # print(logits[0:20])
        return logits


class EarlyStopping():
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}", flush=True)
            if self.counter >= self.patience:
                print('INFO: Early stopping', flush=True)
                self.early_stop = True

def compute_class_weights(sample_counts, cap=50):
    max_count = max(sample_counts)
    B = [max(max_count / c, cap) for c in sample_counts]
    B = np.array(B)
    weights = (B / B.sum())*B
    return torch.tensor(weights, dtype=torch.float32)

if __name__=='__main__':

    random.seed(0)
    np.random.seed(0)  # numpy random generator

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    # Load data
    # load_data(args)

    # Load model
    if args.version == 'noversion':
        ckpt_path = args.model_path
        key = None
    else:
        ckpt_path = './models/models.ckpt'
        if args.output_type == 'cell':
            if args.version == 'ce':
                key = 'cell'
            elif args.version == 'rde':
                key = 'rde'
            else:
                raise ValueError('No version found')
        elif args.output_type == 'gene':
            key = 'gene'
        elif args.output_type == 'gene_batch':
            key = 'gene'
        elif args.output_type == 'gene_expression':  # Not recommended
            key = 'gene'
        else:
            raise ValueError('output_mode must be one of cell gene, gene_batch, gene_expression')


    train_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_train{args.fold_idx}.h5ad'
    # val_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_train{args.fold_idx}.h5ad'
    val_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_val{args.fold_idx}.h5ad'
    test_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_test{args.fold_idx}.h5ad'

    if args.data_name == 'ms':
        celltype_key = 'celltype'
    elif args.data_name == 'COVID':
        celltype_key = 'cell_type'
    elif args.data_name == 'NSCLC':
        celltype_key = 'cell_type'

    train_adata = sc.read_h5ad(train_datapath)
    val_adata = sc.read_h5ad(val_datapath)
    test_adata = sc.read_h5ad(test_datapath)

    unique_labels = np.unique(train_adata.obs[celltype_key])
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    train_dataset = SCDataset(train_adata, gene_list,label_to_int)
    val_dataset = SCDataset(val_adata, gene_list,label_to_int)
    test_dataset = SCDataset(test_adata, gene_list,label_to_int)

    train_num = len(train_dataset)
    train_class_num = np.unique(train_dataset.label, return_counts=True)[1]
    n_class = len(np.unique(train_dataset.label))
    print(f"n_class: {n_class}", flush=True)



    # sample_weights = 1.0 / train_class_num[train_dataset.label]
    # sample_weights = sample_weights / np.sum(sample_weights)
    # print(f"sample_weights: {sample_weights}", flush=True)
    #
    # class_num = np.unique(train_dataset.label, return_counts=True)[1].tolist()
    # print(class_num)
    # class_weight = compute_class_weights(class_num)
    # print(f"class_weight: {class_weight}", flush=True)

    # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, train_num, replacement=True)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(len(train_loader), flush=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    print(len(val_loader), flush=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = LinearProbingClassifier(ckpt_path=ckpt_path,prompt_settings = prompt_settings,key= key,n_class=n_class,pool_type = args.pool_type)
    model.build()
    model = model.to(device)

    print("-" * 89, flush=True)
    learnable_params = {k: v for k, v in model.named_parameters() if v.requires_grad == True}
    for k, v in learnable_params.items():
        print(f"Learnable params {k} with shape {v.shape}", flush=True)
    # post_freeze_param_count = sum(
    #     dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    # print("Total Pre freeze Params: %.2fM" % (pre_freeze_param_count / 1e6,))
    # print("Total Post freeze Params: %.2fM" % (post_freeze_param_count / 1e6,))
    print("-" * 89, flush=True)


    # loss_fn = nn.CrossEntropyLoss(weight=class_weight).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = CosineAnnealingWarmupRestarts(
    #     optimizer,
    #     first_cycle_steps=15,
    #     cycle_mult=2,
    #     max_lr=LEARNING_RATE,
    #     min_lr=1e-7,
    #     warmup_steps=5,
    #     gamma=0.9
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=5 ,min_lr=1e-7)  #0.9 10

    print('start training...', flush=True)
    bestmodel = None
    max_acc = 0.0
    best_val_loss = float("inf")
    early_stopping = EarlyStopping()
    UNASSIGN_THRES = 0.0
    softmax = nn.Softmax(dim=-1)
    for i in range(args.epoch):
        model.train()
        # print(model, flush=True)
        running_loss = 0.0
        cum_acc = 0.0
        predictions_train = []
        truths_train = []
        # print('start training...', flush=True)
        for index, (data, labels) in enumerate(train_loader):
            index += 1
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            loss = loss_fn(logits, labels)
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            # optimizer.zero_grad()
            running_loss += loss.item()

            final = softmax(logits)
            final = final.argmax(dim=-1)
            pred_num = labels.size(0)
            truths_train.extend(labels)
            predictions_train.extend(final)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * cum_acc / len(train_loader)
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==', flush=True)

        # print(torch.stack(predictions_train).detach().cpu().numpy().tolist())
        # print(torch.stack(truths_train).detach().cpu().numpy().tolist())

        # scheduler.step()
        # print(classification_report(torch.stack(truths_train).detach().cpu().numpy(), torch.stack(predictions_train).detach().cpu().numpy()), flush=True)

        # for name, param in model.named_parameters():
        #     if "encoder.transformer_encoder.0.Space_Adapter.D_fc1.weight" in name:
        #         print(f"{name}:\n{param.data}")
        #     if "fc1.0.weight" in name:
        #         print(f"{name}:\n{param.data}")

        model.eval()
        # print(model, flush=True)
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            # print(len(val_loader), flush=True)
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                val_loss = loss_fn(logits, labels_v)
                running_loss += val_loss.item()
                # softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                # final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.extend(final)
                # print("val prediction")
                # print(final)
                truths.extend(labels_v)
                # print("val truths")
                # print(labels_v)
                # tqdm.write(f'Batch {index + 1}/{len(val_loader)}, Loss: {val_loss.item()}')
            epoch_valloss = running_loss / len(val_loader)
            scheduler.step(epoch_valloss)
            del data_v, labels_v, logits, final_prob, final

            # for name, param in model.named_parameters():
            #     if "encoder.transformer_encoder.0.Space_Adapter.D_fc1.weight" in name:
            #         print(f"{name}:\n{param.data}")
            #     if "fc1.0.weight" in name:
            #         print(f"{name}:\n{param.data}")

            # print(truths, flush=True)
            # print(predictions, flush=True)
            # no_drop = predictions != -1
            predictions = torch.stack(predictions).cpu().numpy()
            truths = torch.stack(truths).cpu().numpy()

            # print(predictions.tolist())
            # print(truths.tolist())

            cur_acc = balanced_accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            accuracy = accuracy_score(truths, predictions)

            print(
                f'    ==  Epoch: {i} | Validation Loss: {epoch_valloss:.6f} | F1 Score: {f1:.6f} | Accuracy: {accuracy:.6f}  ==',
                flush=True)

            if epoch_valloss < best_val_loss:
                best_val_loss = epoch_valloss
                best_model = copy.deepcopy(model)
                print(f"Best model with loss {best_val_loss:5.4f}", flush=True)
            early_stopping(epoch_valloss)

            print(classification_report(truths, predictions), flush=True)
            # print(truths, flush=True)
            # print(predictions, flush=True)
            if early_stopping.early_stop:
                break


    torch.save(best_model.state_dict(),
               ckpt_dir + args.data_name + '_' + args.peft_type + '_' + args.fold_idx+ '_' + args.pre_normalized + '_' + f"best_model.pt")
    print('Best model saved successfully!', flush=True)
    del predictions, truths


    def test(model: nn.Module, test_loader: DataLoader) -> float:
        model.eval()
        predictions = []
        truths = []
        for index, (data_t, labels_t) in enumerate(test_loader):
            data_t, labels_t = data_t.to(device), labels_t.to(device)
            logits = model(data_t)
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            # final[np.amax(np.array(final_prob.detach().cpu().numpy()), axis=-1) < UNASSIGN_THRES] = -1
            predictions.append(final.detach().cpu().numpy())
            truths.append(labels_t.detach().cpu().numpy())
            # tqdm.write(f'Batch {index + 1}/{len(test_loader)}')
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)
        # save_dict = {
        #     "labels": truths,
        #     "results": predictions,
        #     "id_maps": label_dict
        # }
        # with open(ckpt_dir + f"{data_name}_{prompt_type}_{str(fold_idx)}_results.pkl", "wb") as f:
        #     pickle.dump(save_dict, f)
        return predictions, truths


    predictions, celltypes_labels = test(best_model, test_loader)
    from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score,accuracy_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    print(f"accuracy: {accuracy:.3f}",flush=True)

    balanced_accuracy = balanced_accuracy_score(celltypes_labels, predictions)
    f1 = f1_score(celltypes_labels, predictions, average="macro")
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")

    print(
        f"macro Accuracy: {balanced_accuracy:.3f}, macro Precision: {precision:.3f},macro Recall: {recall:.3f}, "f"macro F1: {f1:.3f}",
        flush=True)
    micro_f1 = f1_score(celltypes_labels, predictions, average="micro")
    micro_precision = precision_score(celltypes_labels, predictions, average="micro")
    micro_recall = recall_score(celltypes_labels, predictions, average="micro")
    print(
        f"micro Accuracy: {balanced_accuracy:.3f}, micro Precision: {micro_precision:.3f},micro Recall: {micro_recall:.3f}, "f"micro F1: {micro_f1:.3f}",
        flush=True)








