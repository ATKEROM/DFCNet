

from torch import softmax

from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)





class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)



class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc.weight, gain=nn.init.calculate_gain('linear'))
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)

    def compute_loss(self, predictions, label):
        return self.loss_fn(predictions, label)



# class CFB(nn.Module):
#     def __init__(self, CFB_K, CFB_O, DROPOUT_R, img_feat_size, ques_feat_size,is_first):
#         super(CFB, self).__init__()
#         self.CFB_K = CFB_K
#         self.CFB_O = CFB_O
#         self.is_first = is_first
#         self.proj_i = nn.Sequential(
#             nn.Linear(img_feat_size, 2 * CFB_K * CFB_O),
#             nn.LayerNorm(2 * CFB_K * CFB_O),
#             nn.ReLU(inplace=True)
#         )
#         self.proj_q = nn.Sequential(
#             nn.Linear(ques_feat_size, 2 * CFB_K * CFB_O),
#             nn.LayerNorm(2 * CFB_K * CFB_O),
#             nn.ReLU(inplace=True)
#         )
#         self.dropout = nn.Dropout(DROPOUT_R)
#         self.pool = nn.AvgPool1d(CFB_K, stride=CFB_K)
#         self.cross_attention = Transformer(2 * CFB_K * CFB_O, 1, 3, 100, 64, dropout=0.2)
#         self.fc = nn.Linear(2 * CFB_O, CFB_O)

#     def forward(self, img_feat, ques_feat, exp_in=1):

#         batch_size = img_feat.shape[0]
#         img_feat = self.proj_i(img_feat)
#         ques_feat = self.proj_q(ques_feat)
#         self.attn = torch.stack(
#             (img_feat,ques_feat),
#             dim=1
#         )
#         self.attn_output = self.cross_attention(self.attn)
#         img_feat = img_feat + self.attn_output[:, 0, :]
#         ques_feat = ques_feat + self.attn_output[:, 1, :]
#         exp_out = torch.mul(img_feat, ques_feat)
#         exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)
#         self.is_first = False
#         z = self.pool(exp_out) * self.CFB_K
#         z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
#         z = F.normalize(z.view(batch_size, -1))
#         z =self.fc(z)
#         return z, exp_out


class SNNBranch(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum, hidden_size=None):
        super(SNNBranch, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(neuron.IFNode())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(neuron.IFNode())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class CFB(nn.Module):
    def __init__(self, CFB_K, CFB_O, DROPOUT_R, img_feat_size, ques_feat_size, is_first,
                 use_cross_attention=True,    # 是否启用交叉注意力层
                 use_matrix_factorization=True):  # 是否启用矩阵分解
        super(CFB, self).__init__()
        self.CFB_K = CFB_K
        self.CFB_O = CFB_O
        self.is_first = is_first
        self.use_cross_attention = use_cross_attention
        self.use_matrix_factorization = use_matrix_factorization

        # 矩阵分解消融：条件初始化投影层（新增注释）
        if self.use_matrix_factorization:
            self.proj_i = nn.Sequential(
                nn.Linear(img_feat_size, 2 * CFB_K * CFB_O),
                nn.LayerNorm(2 * CFB_K * CFB_O),
                nn.ReLU(inplace=True)
            )
            self.proj_q = nn.Sequential(
                nn.Linear(ques_feat_size, 2 * CFB_K * CFB_O),
                nn.LayerNorm(2 * CFB_K * CFB_O),
                nn.ReLU(inplace=True)
            )
            # 基线模式：池化后维度为 2*CFB_O（新增注释）
            fc_in_features = 2 * CFB_O
        else:
            self.proj_i = nn.Sequential(
                nn.Linear(img_feat_size, CFB_O),  # 简化为直接映射到CFB_O
                nn.LayerNorm(CFB_O),
                nn.ReLU(inplace=True)
            )
            self.proj_q = nn.Sequential(
                nn.Linear(ques_feat_size, CFB_O),
                nn.LayerNorm(CFB_O),
                nn.ReLU(inplace=True)
            )
            # 消融矩阵分解模式：池化后维度为 CFB_O/CFB_K（新增注释）
            # 假设CFB_O能被CFB_K整除（如300/5=60）
            fc_in_features = CFB_O // CFB_K

        self.dropout = nn.Dropout(DROPOUT_R)

        # 注意力消融：条件初始化交叉注意力层（新增注释）
        if self.use_cross_attention:
            # 适配消融后的投影层维度（新增注释）
            cross_attn_dim = 2 * CFB_K * CFB_O if use_matrix_factorization else CFB_O
            self.cross_attention = Transformer(cross_attn_dim, 1, 3, 100, 64, dropout=0.2)

        self.pool = nn.AvgPool1d(CFB_K, stride=CFB_K)
        # 动态设置全连接层输入维度（修复错误关键）（新增注释）
        self.fc = nn.Linear(fc_in_features, CFB_O)

    def forward(self, img_feat, ques_feat, exp_in=1):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)
        ques_feat = self.proj_q(ques_feat)

        if self.use_cross_attention:
            self.attn = torch.stack((img_feat, ques_feat), dim=1)
            self.attn_output = self.cross_attention(self.attn)
            img_feat = img_feat + self.attn_output[:, 0, :]
            ques_feat = ques_feat + self.attn_output[:, 1, :]

        exp_out = torch.mul(img_feat, ques_feat)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)
        self.is_first = False

        # 调整维度以适配AvgPool1d（新增注释）
        exp_out_reshaped = exp_out.unsqueeze(1)  # 形状变为 (batch_size, 1, feature_dim)
        
        z = self.pool(exp_out_reshaped) * self.CFB_K
        z = z.squeeze(1)  # 恢复为 (batch_size, feature_dim)

        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))
        z = self.fc(z)

        return z, exp_out





class LSTM(nn.Module):
    def __init__(self,
                 t_size=1,
                 input_size=256,
                 hidden_size=128,
                 output_dim=300,
                 num_layers=2,
                 dropout=0.3):
        super().__init__()

        self.input_adapter = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.ReLU()
        )

        self.bilstm = nn.LSTM(
            input_size=t_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.feature_pool = nn.Sequential(
            nn.Linear(2 * hidden_size, output_dim),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cfb_proj = nn.Linear(hidden_size, output_dim)



    def forward(self, x):
        x = self.input_adapter(x)
        x = x.unsqueeze(-1)
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.transpose(1, 2)
        x = self.feature_pool(x)
        x=x.squeeze(-1)
        x = self.cfb_proj(x)
        return x







class DCFNet(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(DCFNet, self).__init__()


        print('Initializing model variables...', end='')
        self.dim_out = params_model['dim_out']
        self.hidden_size_encoder = params_model['encoder_hidden_size']
        self.hidden_size_decoder = params_model['decoder_hidden_size']
        self.r_enc = params_model['dropout_encoder']
        self.r_proj = params_model['dropout_decoder']
        self.depth_transformer = params_model['depth_transformer']
        self.additional_triplets_loss = params_model['additional_triplets_loss']
        self.reg_loss = params_model['reg_loss']
        self.r_dec = params_model['additional_dropout']
        self.momentum = params_model['momentum']
        self.first_additional_triplet = params_model['first_additional_triplet']
        self.second_additional_triplet = params_model['second_additional_triplet']
        self.num_classes = params_model['num_classes']
        self.alpha=1
        self.entropy_weight = 0.2
        print('Initializing trainable models...', end='')


        print('Defining CFB...', end='')
        self.CFB_K = 5
        self.CFB_O = 300
        self.img_feat_size  = self.CFB_O
        self.ques_feat_size = self.CFB_O

        self.DROPOUT = 0.5

        self.cfb_is_first = True

        self.cfb_audio = CFB(
            CFB_K=self.CFB_K,
            CFB_O=self.CFB_O,
            DROPOUT_R=self.DROPOUT,
            img_feat_size=self.img_feat_size,
            ques_feat_size=self.ques_feat_size,
            is_first=self.cfb_is_first
        )

        self.cfb_video = CFB(
            CFB_K=self.CFB_K,
            CFB_O=self.CFB_O,
            DROPOUT_R=self.DROPOUT,
            img_feat_size=self.img_feat_size,
            ques_feat_size=self.ques_feat_size,
            is_first=self.cfb_is_first
        )
        print('Done')


        self.audio_enc = EmbeddingNet(
            input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=self.CFB_O,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

        self.video_enc = EmbeddingNet(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=self.CFB_O,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

        self.cross_attention = Transformer(self.CFB_O, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)


        self.W_proj = EmbeddingNet(
            input_size=300,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        self.D = EmbeddingNet(
            input_size=self.dim_out,
            output_size=300,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )


        self.lstm_audio = LSTM(
            input_size=input_size_audio,
            output_dim=self.CFB_O,
            num_layers=2,
            dropout=0.3
        )

        self.lstm_video = LSTM(
            input_size=input_size_video,
            output_dim=self.CFB_O,
            num_layers=2,
            dropout=0.3
        )



        self.audio_proj = EmbeddingNet(input_size=self.CFB_O, hidden_size=self.hidden_size_decoder, output_size=self.dim_out,
                                   dropout=self.r_proj, momentum=self.momentum, use_bn=True)
        self.video_proj = EmbeddingNet(input_size=self.CFB_O, hidden_size=self.hidden_size_decoder, output_size=self.dim_out,
                                   dropout=self.r_proj, momentum=self.momentum, use_bn=True)
        self.audio_rec = EmbeddingNet(input_size=self.dim_out, output_size=self.CFB_O, dropout=self.r_dec, momentum=self.momentum,
                                  use_bn=True)
        self.video_rec = EmbeddingNet(input_size=self.dim_out, output_size=self.CFB_O, dropout=self.r_dec, momentum=self.momentum,
                                  use_bn=True)


        print('Defining classifier...', end='')
        self.a_classifier = ClassificationHead(input_dim=self.CFB_O, num_classes=self.num_classes)
        self.v_classifier = ClassificationHead(input_dim=self.CFB_O, num_classes=self.num_classes)
        print('Done')

        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, self.CFB_O))


        print('Defining optimizers...', end='')
        self.lr = params_model['lr']

        params_group_1 = [
            {'params': self.audio_proj.parameters(), 'lr': 5e-3, 'weight_decay': 1e-5},
            {'params': self.video_proj.parameters(), 'lr': 5e-4, 'weight_decay': 1e-5},
        ]

        params_group_2 = [
            {'params': self.audio_rec.parameters(), 'lr': 3e-3, 'weight_decay': 1e-5},
            {'params': self.video_rec.parameters(), 'lr': 3e-4, 'weight_decay': 1e-5},
        ]

        params_group_3 = [
            {'params': self.audio_enc.parameters(), 'lr': 2e-3, 'weight_decay': 1e-5},
            {'params': self.video_enc.parameters(), 'lr': 2e-4, 'weight_decay': 1e-5},
            {'params': self.lstm_audio.parameters(), 'lr': 5e-3, 'weight_decay': 1e-5},
            {'params': self.lstm_video.parameters(), 'lr': 5e-4, 'weight_decay': 1e-5},
        ]

        params_group_4 = [
            {'params': self.D.parameters(), 'lr': 5e-3, 'weight_decay': 1e-5},
            {'params': self.W_proj.parameters(), 'lr': 5e-3, 'weight_decay': 1e-5},
            {'params': self.a_classifier.parameters(), 'lr': 5e-3, 'weight_decay': 1e-5},
            {'params': self.v_classifier.parameters(), 'lr': 5e-3, 'weight_decay': 1e-5},
            {'params': self.cross_attention.parameters(), 'lr': 5e-5, 'weight_decay': 1e-5},

        ]

        params_group_5 = [
            {'params': self.cfb_audio.parameters(), 'lr': 7e-4, 'weight_decay': 1e-5},
            {'params': self.cfb_video.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
        ]

        self.optimizer_gen = optim.Adam(
            params_group_1 + params_group_2 + params_group_3 + params_group_4 +params_group_5 ,
            lr=self.lr,
            weight_decay=1e-4
        )

        self.optimizer = optim.Adam(
             params_group_5,
            lr=self.lr,
            weight_decay=1e-4
        )

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3, verbose=True)
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)
        print('Done')
        # SNN 分支的循环次数，用于重复执行神经元网络
        self.T = 10

        print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        self.criterion=nn.CrossEntropyLoss()
        print('Done')

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)
        self.lr_scheduler.step(value)

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):


        self.phi_a = self.audio_enc(audio)
        self.phi_v = self.video_enc(image)


        self.phi_a1 = self.lstm_audio(audio)
        self.phi_v1 = self.lstm_video(image)






        self.phi_a_cfb, _ = self.cfb_audio(self.phi_a, self.phi_a)
        self.phi_a1_cfb, _ = self.cfb_audio(self.phi_a1, self.phi_a1)





        cfb_output_a, _ = self.cfb_audio(self.phi_a, self.phi_a1)
        cfb_output_v, _ = self.cfb_video(self.phi_v, self.phi_v1)



        cfb_output_a = self.phi_a_cfb + self.phi_a1_cfb + cfb_output_a
        cfb_output_v= self.phi_v +  self.phi_v1 + cfb_output_v



        self.a_emb = cfb_output_a
        self.v_emb = cfb_output_v


        self.a_logits = self.a_classifier(self.a_emb)
        self.v_logits = self.v_classifier(self.v_emb)

        self.a = self.a_emb
        self.v = self.v_emb


        score_a = sum([
            softmax(self.a_logits, dim=0)[i, self.label[i]]
            for i in range(self.a_logits.size(0))
        ])


        score_v = sum([
            softmax(self.v_logits, dim=0)[i, self.label[i]]
            for i in range(self.v_logits.size(0))
        ])


        self.ratio_v = score_v / score_a
        self.ratio_a = 1 / self.ratio_v



        self.phi_a_neg = self.audio_enc(negative_audio)
        self.phi_v_neg = self.video_enc(negative_image)


        self.phi_a_neg1 = self.lstm_audio(negative_audio)
        self.phi_v_neg1 = self.lstm_video(negative_image)

 

        self.phi_a_neg_cfb, _ = self.cfb_audio(self.phi_a_neg, self.phi_a_neg)
        self.phi_a_neg1_cfb, _ = self.cfb_audio(self.phi_a_neg1, self.phi_a_neg1)




        cfb_output_a_neg, _ = self.cfb_audio(self.phi_a_neg, self.phi_a_neg1)
        cfb_output_v_neg, _ = self.cfb_video(self.phi_v_neg, self.phi_v_neg1)





        cfb_output_a_neg= self.phi_a_neg_cfb + self.phi_a_neg1_cfb + cfb_output_a_neg
        cfb_output_v_neg= self.phi_v_neg + self.phi_v_neg1 + cfb_output_v_neg


        self.w = word_embedding
        self.w_neg = negative_word_embedding
        self.theta_w = self.W_proj(word_embedding)

        self.theta_w_neg = self.W_proj(negative_word_embedding)



        self.rho_w = self.D(self.theta_w)
        self.rho_w_neg = self.D(self.theta_w_neg)



        self.positive_input = torch.stack(
            (cfb_output_a + self.pos_emb1D[0, :], cfb_output_v + self.pos_emb1D[1, :]),
            dim=1
        )


        self.negative_input = torch.stack(
            (cfb_output_a_neg + self.pos_emb1D[0, :], cfb_output_v_neg + self.pos_emb1D[1, :]),
            dim=1
        )



        self.phi_attn = self.cross_attention(self.positive_input)
        self.phi_attn_neg = self.cross_attention(self.negative_input)


        self.audio_fe_attn = self.phi_a + self.phi_attn[:, 0, :]
        self.video_fe_attn = self.phi_v + self.phi_attn[:, 1, :]

        self.audio_fe_neg_attn = self.phi_a_neg + self.phi_attn_neg[:, 0, :]
        self.video_fe_neg_attn = self.phi_v_neg + self.phi_attn_neg[:, 1, :]


        self.theta_v = self.video_proj(self.video_fe_attn)
        self.theta_v_neg = self.video_proj(self.video_fe_neg_attn)


        self.theta_a = self.audio_proj(self.audio_fe_attn)
        self.theta_a_neg = self.audio_proj(self.audio_fe_neg_attn)



        self.phi_v_rec = self.video_rec(self.theta_v)
        self.phi_a_rec = self.audio_rec(self.theta_a)


        self.se_em_hat1 = self.audio_proj(self.phi_a_rec)
        self.se_em_hat2 = self.video_proj(self.phi_v_rec)





        self.rho_a = self.D(self.theta_a)
        self.rho_a_neg = self.D(self.theta_a_neg)
        self.rho_v = self.D(self.theta_v)
        self.rho_v_neg = self.D(self.theta_v_neg)



        # print('Finished forward pass.')




    def backward(self, optimize):

        if self.additional_triplets_loss == True:
            first_pair = self.first_additional_triplet * (
                    self.triplet_loss(self.theta_a, self.theta_w, self.theta_a_neg) +
                    self.triplet_loss(self.theta_v, self.theta_w, self.theta_v_neg)
            )
            second_pair = self.second_additional_triplet * (
                    self.triplet_loss(self.theta_w, self.theta_a, self.theta_w_neg) +
                    self.triplet_loss(self.theta_w, self.theta_v, self.theta_w_neg)
            )
            l_t = first_pair + second_pair


        if self.reg_loss == True:
            l_r = (
                    self.criterion_reg(self.phi_v_rec, self.phi_v) +
                    self.criterion_reg(self.phi_a_rec, self.phi_a) +
                    self.criterion_reg(self.theta_v, self.theta_w) +
                    self.criterion_reg(self.theta_a, self.theta_w)
            )


        l_rec = (
                self.criterion_reg(self.w, self.rho_v) +
                self.criterion_reg(self.w, self.rho_a) +
                self.criterion_reg(self.w, self.rho_w)
        )



        l_ctv = self.triplet_loss(self.rho_w, self.rho_v, self.rho_v_neg)
        l_cta = self.triplet_loss(self.rho_w, self.rho_a, self.rho_a_neg)
        l_ct = l_cta + l_ctv
        l_cmd = l_rec + l_ct




        l_tv = self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg)
        l_ta = self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg)
        l_at = self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg)
        l_vt = self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg)
        l_w = l_ta + l_at + l_tv + l_vt


        a_classification_loss = self.criterion(self.a_logits,self.label)
        v_classification_loss = self.criterion(self.v_logits,self.label)

        loss_gen = l_cmd + l_w+a_classification_loss + v_classification_loss
        if self.additional_triplets_loss == True:
            loss_gen += l_t
        if self.reg_loss == True:
            loss_gen += l_r


        if optimize == True:
            if self.epoch >=2 and self.epoch <=30 :
                self.optimizer.zero_grad()


                audio_sim = exp_cosine_euclidean_sim(self.a, self.audio_class)
                visual_sim = exp_cosine_euclidean_sim(self.v, self.visual_class)

                audio_sim = torch.softmax(audio_sim, dim=1)
                visual_sim = torch.softmax(visual_sim, dim=1)

                audio_entropy = compute_entropy(audio_sim)
                visual_entropy = compute_entropy(visual_sim)

                loss_entropy = self.entropy_weight * (audio_entropy + visual_entropy)
                loss_class_v = self.criterion(visual_sim, self.label)
                loss_class_a = self.criterion(audio_sim, self.label)
                ratio = self.ratio_a + self.ratio_v



                if self.ratio_a > 1:
                    beta = 0
                    lam = 1 * self.alpha * ratio / self.ratio_v
                elif self.ratio_a < 1:
                    beta = 1 * self.alpha * ratio / self.ratio_a
                    lam = 0
                loss_ad = beta * loss_class_a + lam * loss_class_v +loss_entropy
                loss_ad.backward(retain_graph=True)
                self.optimizer.step()
            else:

                self.optimizer_gen.zero_grad()
                loss_gen.backward(retain_graph=True)
                self.optimizer_gen.step()


        loss = {
            'aut_enc': 0,
            'gen_cyc': 0,
            'gen_reg': 0,
            'gen': loss_gen - a_classification_loss - v_classification_loss,
        }
        loss_numeric = loss['gen_cyc'] + loss['gen']
        return loss_numeric, loss


    def optimize_params(self, audio, video, cls_numeric, cls_embedding, audio_negative, video_negative,
                        negative_cls_embedding, audio_class, visual_class, epoch,  label,  optimize=False):

        self.label = label
        self.audio_class = audio_class
        self.visual_class = visual_class
        self.epoch = epoch
        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)
        loss_numeric, loss = self.backward(optimize)

        return loss_numeric, loss








    def get(self, audio, video):

        # 对音频和视频进行编码
        phi_a = self.audio_enc(audio)
        phi_v = self.video_enc(video)

        phi_a1 = self.lstm_audio(audio)
        phi_v1 = self.lstm_video(video)
    



        phi_a_cfb, _ = self.cfb_audio(phi_a, phi_a)
        phi_a1_cfb, _ = self.cfb_audio(phi_a1, phi_a1)




        cfb_output_a, _ = self.cfb_audio(phi_a, phi_a1)


        cfb_output_v, _ = self.cfb_video(phi_v, phi_v1)


        cfb_output_a = phi_a_cfb +  phi_a1_cfb+ cfb_output_a
        cfb_output_v = phi_v + phi_v1+ cfb_output_v



        input_concatenated = torch.stack(
            (cfb_output_a + self.pos_emb1D[0, :], cfb_output_v + self.pos_emb1D[1, :]),
            dim=1
        )



        phi_attn = self.cross_attention(input_concatenated)
  

        phi_a = phi_a + phi_attn[:, 0, :]
        phi_v = phi_v + phi_attn[:, 1, :]


        a_emb = phi_a
        v_emb = phi_v
        return a_emb,v_emb




    def get_embeddings(self, audio, video, embedding):


        phi_a = self.audio_enc(audio)
        phi_v = self.video_enc(video)
        theta_w = self.W_proj(embedding)

        phi_a1 = self.lstm_audio(audio)
        phi_v1 = self.lstm_video(video)




        phi_a_cfb, _ = self.cfb_audio(phi_a, phi_a)
        phi_a1_cfb, _ = self.cfb_audio(phi_a1, phi_a1)


        cfb_output_a, _ = self.cfb_audio(phi_a, phi_a1)
        cfb_output_v, _ = self.cfb_video(phi_v, phi_v1)


        cfb_output_a = phi_a_cfb + phi_a1_cfb + cfb_output_a
        cfb_output_v = phi_v + phi_v1 + cfb_output_v


        input_concatenated = torch.stack(
            (cfb_output_a + self.pos_emb1D[0, :], cfb_output_v + self.pos_emb1D[1, :]),
            dim=1
        )

 

        phi_attn = self.cross_attention(input_concatenated)



        phi_a = phi_a + phi_attn[:, 0, :]
        phi_v = phi_v + phi_attn[:, 1, :]


        theta_v = self.video_proj(phi_v)
        theta_a = self.audio_proj(phi_a)



        return theta_a, theta_v, theta_w





def exp_cosine_euclidean_sim(x, proto):

    x = F.normalize(x, p=2, dim=-1)
    proto = F.normalize(proto, p=2, dim=-1)

    cos_sim = torch.mm(x, proto.t())

    eu_dist = torch.cdist(x, proto, p=2)

    scaled_eu = eu_dist / (2 ** 0.5)

    return torch.exp(cos_sim - scaled_eu)



def compute_entropy(similarity_matrix):
    epsilon = 1e-10
    softmax_sim = F.softmax(similarity_matrix, dim=1) + epsilon
    log_softmax_sim = F.log_softmax(similarity_matrix, dim=1)
    entropy = torch.mean(log_softmax_sim)
    return entropy

