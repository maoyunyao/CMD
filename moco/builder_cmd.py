import torch
import torch.nn as nn
import torch.nn.functional as F

from .GRU import BIGRU

def loss_kld(inputs, targets):
    inputs = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')

class MoCo(nn.Module):
    def __init__(self, skeleton_representation, args_bi_gru, dim=128, K=65536, m=0.999, T=0.07,
                 teacher_T=0.05, student_T=0.1, cmd_weight=1.0, topk=1024, mlp=False, pretrain=True):
        super(MoCo, self).__init__()
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_q_motion = BIGRU(**args_bi_gru)
            self.encoder_q_bone = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_q_motion)
            weights_init_gru(self.encoder_q_bone)
        else:
            self.K = K
            self.m = m
            self.T = T
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.cmd_weight = cmd_weight
            self.topk = topk
            mlp=mlp
            print(" MoCo parameters",K,m,T,mlp)
            print(" CMD parameters: teacher-T %.2f, student-T %.2f, cmd-weight: %.2f, topk: %d"%(teacher_T,student_T,cmd_weight,topk))
            print(skeleton_representation)


            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)
            self.encoder_q_motion = BIGRU(**args_bi_gru)
            self.encoder_k_motion = BIGRU(**args_bi_gru)
            self.encoder_q_bone = BIGRU(**args_bi_gru)
            self.encoder_k_bone = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_q_motion)
            weights_init_gru(self.encoder_q_bone)
            weights_init_gru(self.encoder_k)
            weights_init_gru(self.encoder_k_motion)
            weights_init_gru(self.encoder_k_bone)

            #projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_k.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                            nn.ReLU(),
                                                            self.encoder_q_motion.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                            nn.ReLU(),
                                                            self.encoder_k_motion.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_q_bone.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_k_bone.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        self.queue_motion[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_motion[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        self.queue_bone[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_bone[0] = ptr


    def forward(self, im_q, im_k=None, view='joint', knn_eval=False):
        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        # Permute and Reshape
        N, C, T, V, M = im_q.size()
        im_q = im_q.permute(0,2,3,1,4).reshape(N,T,-1)
        im_q_motion = im_q_motion.permute(0,2,3,1,4).reshape(N,T,-1)
        im_q_bone = im_q_bone.permute(0,2,3,1,4).reshape(N,T,-1)

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion, knn_eval)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone, knn_eval)
            elif view == 'all':
                return (self.encoder_q(im_q, knn_eval) + \
                        self.encoder_q_motion(im_q_motion, knn_eval) + \
                            self.encoder_q_bone(im_q_bone, knn_eval)) / 3.
            else:
                raise ValueError        
        
        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # Permute and Reshape
        im_k = im_k.permute(0,2,3,1,4).reshape(N,T,-1)
        im_k_motion = im_k_motion.permute(0,2,3,1,4).reshape(N,T,-1)
        im_k_bone = im_k_bone.permute(0,2,3,1,4).reshape(N,T,-1)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # MOCO
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # CMD loss
        lk_neg = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        lk_neg_motion = torch.einsum('nc,ck->nk', [k_motion, self.queue_motion.clone().detach()])
        lk_neg_bone = torch.einsum('nc,ck->nk', [k_bone, self.queue_bone.clone().detach()])

        # Top-k
        lk_neg_topk, topk_idx = torch.topk(lk_neg, self.topk, dim=-1)
        lk_neg_motion_topk, motion_topk_idx = torch.topk(lk_neg_motion, self.topk, dim=-1)
        lk_neg_bone_topk, bone_topk_idx = torch.topk(lk_neg_bone, self.topk, dim=-1)

        loss_cmd = loss_kld(torch.gather(l_neg_motion, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg_bone, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg, -1, motion_topk_idx) / self.student_T, lk_neg_motion_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg_bone, -1, motion_topk_idx) / self.student_T, lk_neg_motion_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg, -1, bone_topk_idx) / self.student_T, lk_neg_bone_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg_motion, -1, bone_topk_idx) / self.student_T, lk_neg_bone_topk / self.teacher_T)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits, logits_motion, logits_bone, labels, loss_cmd * self.cmd_weight