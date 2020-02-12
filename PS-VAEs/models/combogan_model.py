import numpy as np
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import cv2
from torch.autograd import Variable


class ComboGANModel(BaseModel):
    def name(self):
        return 'ComboGANModel'

    def __init__(self, opt):
        super(ComboGANModel, self).__init__(opt)
        self.opt = opt
        self.count = 0
        self.correct = 0
        self.epoch = -1
        self.train_domain = 1
        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None
        self.ax, self.ay = None, None
        self.bx, self.by = None, None
        self.error_list = None
        self.feature_list = None
        self.target_list = None
        classes = 32
        if opt.est_joint:
            classes = 18
        if opt.est_mnist:
            classes = 10
        if opt.est_mult:
            classes = 50
        self.start_pose_epoch = opt.start_pose_epoch
        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.back_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.back_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.fake_B = torch.zeros_like(self.real_A)
        self.fake_A = torch.zeros_like(self.real_B)
        self.rec_A = torch.zeros_like(self.real_A)
        self.rec_B = torch.zeros_like(self.real_B)

        self.cyc_A = torch.zeros_like(self.real_A)
        self.cyc_B = torch.zeros_like(self.real_B)

        self.idt_A = torch.zeros_like(self.real_A)
        self.idt_B = torch.zeros_like(self.real_B)

        self.mask_A = torch.zeros_like(self.real_A)
        self.mask_B = torch.zeros_like(self.real_B)


        self.gt_B = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids, classes, opt.divide_dims, opt.oldmodel)
        # False if datasets include NaN
        self.nanflag = True

        #blur_fn = lambda x : torch.nn.functional.conv2d(x, self.Tensor(util.gkern_2d()), groups=3, padding=2)
        blur_fn = lambda x,n : torch.nn.functional.conv2d(x, self.Tensor(util.gkern_2d(channel=n)), groups=n, padding=2)
        self.netD = networks.define_D(opt.input_nc, opt.output_nc, opt.ndf, opt.netD_n_layers, self.n_domains + (self.start_pose_epoch >= 3), blur_fn, opt.norm, opt.spectral_norm, self.gpu_ids, opt.divide_dims)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if not self.isTrain and self.opt.discriminate:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            self.fake_labelpools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.L1 = torch.nn.SmoothL1Loss()
            self.heatmap_dis = torch.nn.SmoothL1Loss(reduction='sum')
            self.BCEloss = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='sum')
            self.crossentropy = torch.nn.CrossEntropyLoss()
            self.cosinedistance = torch.nn.CosineSimilarity()

            self.downsample = torch.nn.AvgPool2d(32, stride=24)
            self.criterionCycle = self.L1
            self.criterionFor = lambda y,t : self.L1(self.downsample(y), self.downsample(t))
            #self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t))
            self.criterionIdt = self.L1
            #self.criterionLatent = lambda y,t : torch.mean(1 - self.cosinedistance(y, t.detach()))
            self.criterionLatent = lambda y,t : self.L1(y,t.detach())
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor,flatloss=opt.flatloss)
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle, self.loss_idt = [0]*self.n_domains, [0]*self.n_domains
            self.loss_pose, self.loss_lat = [0]*self.n_domains, [0]*self.n_domains
            self.loss_labelclustring = [0]*self.n_domains
            self.loss_kl = [0]*self.n_domains
            # initialize loss multipliers
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward
            self.lambda_kl = opt.lambda_kl
            self.lambda_pose = opt.lambda_pose

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input,epoch=-1,val=False):
        #print(input['A'].size(),input['B'].size())
        self.epoch = epoch
        input_A = input['A']
        #print(input_A.size()) = (1,3,256,128)
        self.DA = input['DA'][0]
        if self.isTrain and not val:
            #remove lack mask and ground truth
            input_A = input_A[:,:self.opt.input_nc]
            self.real_A.resize_(input_A.size()).copy_(input_A)
        else:
            if self.opt.est_mnist:
                gt_B = input_A[:,-1]
            else:
                gt_B = input_A[:,1]
            self.gt_B.resize_(gt_B.size()).copy_(gt_B)
            #self.real_A.resize_(torch.unsqueeze(input_A[:,0],0).size()).copy_(torch.unsqueeze(input_A[:,0],0))
            self.gt_B = self.gt_B.type(torch.cuda.LongTensor)
            self.real_A.resize_(input_A[:,:self.opt.input_nc].size()).copy_(input_A[:,:self.opt.input_nc])

        if self.isTrain and not val:
            input_B = input['B'][:,:self.opt.output_nc]
            if self.opt.est_joint:
                gt_B = input['B'][:,-2]
            elif self.opt.est_mult:
                gt_B = input['B'][:,-2:]
            else:
                gt_B = input['B'][:,-1]
            #gt_B.size = (B, (2), 256,128)
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.gt_B.resize_(gt_B.size()).copy_(gt_B)
            self.gt_B = self.gt_B.type(torch.cuda.LongTensor)
            self.bx = input['bx'][0]
            self.by = input['by'][0]
            self.DB = input['DB'][0]
        self.image_paths = input['path']
        self.ax = input['ax'][0]
        self.ay = input['ay'][0]

    def test(self, val=False):
        with torch.no_grad():
            dom = self.DA
            #dom = 1
            self.count += self.opt.batchSize
            self.visuals = [self.real_A]
            self.labels = ['real_%d' % dom]
            #d is another domain
            d = 1 - dom
            encoded = self.netG.encode(self.real_A, dom)[0]
            if self.opt.oldmodel:
                fake = self.netG.decode(encoded,d)
                rec = self.netG.decode(self.netG.encode(fake,d)[0],dom)
            else:
                rec = self.netG.decode(self.del_fea(encoded,dom,True))
                fake = self.netG.decode(self.del_fea(encoded,d,True))

            if self.start_pose_epoch == 0:
                label = self.netG.labeldecode(encoded[:,:self.opt.divide_dims], 0)[0]
            elif self.start_pose_epoch == 1:
                label = self.netG.labeldecode(self.netG.labelencode(fake, 0),0)[0]
            else:
                label = self.netG.labeldecode(self.netG.labelencode(self.real_A, 0),0)[0]
            if val:
                if self.opt.est_mnist:
                    pred = np.argmax(label.cpu().float().numpy(),axis=1)
                    true = self.gt_B[:,0,0].cpu().float().numpy()
                    return np.count_nonzero(pred==true) / len(pred)
                elif self.opt.est_joint:
                    pred = label[:1].cpu().float().numpy()
                    true = self.gt_B[:1].cpu().int().numpy()
                    apply_gauss = [cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3) for img in pred[0]]
                    #print(np.max(apply_gauss,axis=(1,2)))
                    pred_joint_pos_2d = [np.unravel_index(np.argmax(joint), joint.shape) for joint in apply_gauss]

                    true_joint_pos_2d = [np.where(true[0]==j) for j in range(1,19)]
                    error_joint_pos_2d = np.array([np.sqrt(np.square(pred[0] - true[0]) + np.square(pred[1] - true[1])) for pred, true in zip(pred_joint_pos_2d, true_joint_pos_2d)])
                    return np.count_nonzero(error_joint_pos_2d <= 20) / 18

            if self.opt.est_mnist:
                pred = np.argmax(label.cpu().float().numpy(),axis=1)
                true = self.gt_B[:,0,0].cpu().int().numpy()

                if self.error_list is None:
                    self.error_list = np.array([pred,true])
                else:
                    self.error_list = np.vstack((self.error_list,[pred,true]))
                self.correct += np.count_nonzero(pred==true)
                #print(self.correct / self.count)
                map_A = torch.zeros_like(self.real_A[0,:1])
                for i in range(10):
                    map_A[:,i*3:(i+1)*3] = label[0,i]
                label = map_A / torch.max(map_A)
            if self.opt.est_joint:

                    #print(label.size())
                    #print(self.gt_B.size())
                    pred = label[:1].cpu().float().numpy()
                    true = self.gt_B[:1].cpu().int().numpy()

                    apply_gauss = [cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3) for img in pred[0]]
                    #print(np.max(apply_gauss,axis=(1,2)))
                    pred_joint_pos_2d = [np.unravel_index(np.argmax(joint), joint.shape) for joint in apply_gauss]
                    true_joint_pos_2d = [np.where(true[0]==j) for j in range(1,19)]
                    error_joint_pos_2d = np.array([np.sqrt(np.square(pred[0] - true[0]) + np.square(pred[1] - true[1])) for pred, true in zip(pred_joint_pos_2d, true_joint_pos_2d)]).reshape((1,-1))
                    if self.error_list is None:
                        self.error_list = error_joint_pos_2d
                    else:
                        self.error_list = np.vstack((self.error_list,error_joint_pos_2d))

            self.visuals.append( fake )
            self.labels.append( 'fake_%d' % d )
            self.visuals.append( label )
            self.labels.append( 'label_%d' % d )
            self.visuals.append( rec )
            self.labels.append( 'rec_%d' % d )

            if self.opt.discriminate:
                pred_fake = self.netD.forward(self.real_A, self.DA)
                for i, p in enumerate(pred_fake):
                    self.visuals.append( p )
                    self.labels.append( 'disc_'+str(self.DA.item())+'_'+str(i)+'_val:'+str(p.mean().item()) )
                pred_fake = self.netD.forward(fake, d)
                for i, p in enumerate(pred_fake):
                    self.visuals.append( p )
                    self.labels.append( 'disc_'+str(d.item())+'_'+str(i)+'_val:'+str(p.mean().item()))
            self.rec_A = self.real_A
            self.rec_B = fake

            if self.opt.reconstruct:
                rec = self.netG.forward(fake, d, self.DA)
                self.visuals.append( rec )
                self.labels.append( 'rec_%d' % d )

    def get_image_paths(self):
        return self.image_paths
    def get_error_list(self):
        return self.error_list, self.feature_list, self.target_list
    def get_digit_accuracy(self):
        return self.correct * 100 / self.count
    def get_nanflag(self):
        return self.nanflag

    def reparameterize(self, mu, logvar):
        if self.isTrain and self.lambda_kl > 0:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_fun(self, pred, true):
        #pred[0] : joint
        #pred[1] : label
        def loss_joint(pred, true):
            label = torch.transpose(torch.transpose(torch.eye(19)[true],1,3),2,3)[:,1:].type(torch.cuda.FloatTensor)
            kernel = util.gkern_2d(size = 15, channel = 18)
            one_kernel = np.ones_like(kernel)
            gmap = torch.nn.functional.conv2d(label, self.Tensor(one_kernel),groups=18,padding=7)
            gmap = torch.clamp(gmap,0,1)
            gmap = torch.nn.functional.conv2d(gmap,self.Tensor(kernel),groups=18,padding=7)
            gmap = torch.clamp(gmap,0,1)
            return self.heatmap_dis(pred,gmap.detach()) / (18 * 1000)
        if len(pred) == 1:
            pred = pred[0]
        if self.opt.est_mnist:
            label = true[:,0,0]
            #print(pred)
            return self.crossentropy(pred,label)
        elif self.opt.est_joint:
            return loss_joint(pred, true)
        else:
            return self.crossentropy(pred,true)

    def backward_D_basic(self, real, fake, domain):
        #print("D")
        #real = torch.unsqueeze(real[:,1],0)
        #fake = torch.unsqueeze(fake[:,1],0)
        # Real
        pred_real = self.netD.forward(real.detach(), domain)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        if self.start_pose_epoch >= 3:
            label_A = self.fake_labelpools[self.DA].query(self.f_encoded_A)
            label_B = self.fake_labelpools[self.DB].query(self.f_encoded_B)
            self.loss_labelclustring[0] = self.backward_D_basic(label_A, label_B, 2)
            if self.start_pose_epoch == 5:
                self.loss_D[self.DA] = self.loss_labelclustring[0]
                self.loss_D[self.DB] = 0
                return

        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.real_A, fake_A, self.DA)

    def del_fea(self, fea, d, noise=False):
        low = [self.opt.divide_dims, (self.opt.ngf + self.opt.divide_dims)//2]
        high = [(self.opt.ngf + self.opt.divide_dims)//2, self.opt.ngf]
        slice_dim = [list(range(low[1])),list(range(low[0])) + list(range(high[0],high[1]))]
        ixs = torch.arange(self.opt.ngf, dtype=torch.int64).to("cuda:"+str(self.opt.gpu_ids[0]))
        return torch.where((low[0] <= ixs[None, :, None, None]) * (ixs[None, :, None, None] < high[1]), torch.zeros((fea.size(2), fea.size(3))).to("cuda:"+str(self.opt.gpu_ids[0])), fea)

    def backward_G_new(self):

        #num_feature : N
        #divide_dims : M
        #depend_feat : (N - M) / 2 (former A later B)

        if torch.sum(torch.isnan(self.real_A)) + torch.sum(torch.isnan(self.real_B)) > 0:
            print("NaN in datasets")
        mu_A, logvar_A = self.netG.encode(self.real_A)
        encoded_A = self.reparameterize(mu_A, logvar_A)
        mu_B, logvar_B = self.netG.encode(self.real_B)
        encoded_B = self.reparameterize(mu_B, logvar_B)
        #print(encoded_A[:,slice_A].size())
        if self.lambda_kl > 0:
            self.loss_kl[self.DA] = -0.5 * torch.mean(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())
            self.loss_kl[self.DB] = -0.5 * torch.mean(1 + logvar_B - mu_B.pow(2) - logvar_B.exp())
        else:
            self.loss_kl[self.DA] = 0
            self.loss_kl[self.DB] = 0

        encoded_AtoA = self.del_fea(encoded_A, 0)
        encoded_BtoB = self.del_fea(encoded_B, 1)
        mu_AtoB = self.del_fea(mu_A, 1)
        mu_BtoA = self.del_fea(mu_B, 0)

        # Optional identity "autoencode" loss
        if self.lambda_idt > 0:

            # Same encoder and decoder should recreate image
            self.idt_A = self.netG.decode(encoded_AtoA)
            self.loss_idt[self.DA] = self.criterionIdt(self.idt_A, self.real_A)
            self.idt_B = self.netG.decode(encoded_BtoB)
            self.loss_idt[self.DB] = self.criterionIdt(self.idt_B, self.real_B)
        else:
            self.loss_idt[self.DA], self.loss_idt[self.DB] = 0, 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.decode(mu_AtoB)
        #pred_fake_B = self.netD.forward(torch.unsqueeze(self.fake_B[:,1],0), self.DB)
        pred_fake_B = self.netD.forward(self.fake_B, self.DB)
        #print("G")
        self.loss_G[self.DA] = self.criterionGAN(pred_fake_B, True)
        # D_B(G_B(B))
        self.fake_A = self.netG.decode(mu_BtoA)
        pred_fake_A = self.netD.forward(self.fake_A, self.DA)
        #print(self.DA,[x.size() for x in pred_fake])
        self.loss_G[self.DB] = self.criterionGAN(pred_fake_A, True)
        if self.lambda_cyc > 0:
            rec_mu_A, rec_logvar_A = self.netG.encode(self.fake_B)
            rec_encoded_A = self.del_fea(rec_mu_A, 0)
            rec_mu_B, rec_logvar_B = self.netG.encode(self.fake_A)
            rec_encoded_B = self.del_fea(rec_mu_B, 1)

            # Forward cycle loss
            self.rec_A = self.netG.decode(rec_encoded_A)
            self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A)
            #self.loss_cycle[self.DA] = 0
            # Backward cycle loss
            self.rec_B = self.netG.decode(rec_encoded_B)
            #self.loss_cycle[self.DB] = self.loss_fun(self.netG.labeldecode(self.netG.encode(self.rec_B)[0][:,:self.opt.divide_dims]),self.gt_B)
            self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B)
            self.cyc_A = self.rec_A
            self.cyc_B = self.rec_B
        else:
            self.loss_cycle[self.DA] = 0
            self.loss_cycle[self.DB] = 0
            self.rec_A = self.fake_A[:,0,:,:]
            self.rec_B = self.fake_B[:,0,:,:]

        # Optional cycle loss on encoding space
        self.loss_lat[self.DA], self.loss_lat[self.DB] = 0, 0
        if self.lambda_enc > 0:

            self.loss_lat[self.DA] = self.criterionLatent(self.netG.encode(self.netG.decode(mu_AtoB.detach()))[0][:,:self.opt.divide_dims],mu_A[:,:self.opt.divide_dims].detach())
            self.loss_lat[self.DB] = self.criterionLatent(self.netG.encode(self.netG.decode(mu_BtoA.detach()))[0][:,:self.opt.divide_dims],mu_B[:,:self.opt.divide_dims].detach())

        if self.start_pose_epoch == 0:
            #ours
            #label_B = self.netG.labeldecode(self.netG.encode(self.real_B, self.DB)[0][:,:self.opt.divide_dims], self.DA)
            label_B = self.netG.labeldecode(mu_B[:,:self.opt.divide_dims])
            self.rec_B = label_B
            self.loss_pose[self.DB] = self.loss_fun(label_B, self.gt_B)
            self.rec_A = self.netG.labeldecode(mu_A[:,:self.opt.divide_dims])
            self.loss_pose[self.DA] = 0
        else:
            self.loss_pose[self.DA] = 0
            self.loss_pose[self.DB] = 0

        self.back_A = self.real_A
        self.back_B = self.real_B
        self.rec_A = self.rec_A[0]
        self.rec_B = self.rec_B[0]
        self.mask_A = pred_fake_B[-1]
        self.mask_B = pred_fake_A[-1]
        if self.opt.est_mnist and self.start_pose_epoch >= 0:
            map_A = torch.zeros_like(self.real_B[0])
            map_B = torch.zeros_like(self.real_B[0])
            for i in range(10):
                    map_A[:,i*3:(i+1)*3] = self.rec_A[0,i]
                    map_B[:,i*3:(i+1)*3] = self.rec_B[0,i]
            self.rec_A = map_A / torch.max(map_A)
            self.rec_B = map_B / torch.max(map_B)
            self.rec_A[self.rec_A < 0] = 0
            self.rec_B[self.rec_B < 0] = 0
        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_kl[self.DA] + self.loss_kl[self.DB]) * self.lambda_kl + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc + \
                 (self.loss_idt[self.DA] + self.loss_idt[self.DB]) * self.lambda_idt + \
                 (self.loss_lat[self.DA] + self.loss_lat[self.DB]) * self.lambda_enc + \
                 (self.loss_pose[self.DA]*0.5 + self.loss_pose[self.DB]) * self.lambda_pose + \
                  self.loss_labelclustring[self.DA] * self.lambda_fwd
        if self.epoch < 6 and self.start_pose_epoch >= 0:
            loss_G = (self.loss_kl[self.DA] + self.loss_kl[self.DB]) * self.lambda_kl +(self.loss_pose[self.DB]) * self.lambda_pose

        loss_G.backward()

    def optimize_parameters(self):
        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G_new()
        self.netG.step_grads(self.DA, self.DB)
        if self.epoch >= 5:
            # D_A and D_B
            self.netD.zero_grads(self.DA, self.DB)
            self.backward_D()
            self.netD.step_grads(self.DA, self.DB)
        if torch.sum(torch.isnan(self.fake_A)) + torch.sum(torch.isnan(self.fake_B)) > 0 and self.nanflag:
            print("NaN in generated images")
            self.nanflag = False

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle)
        idt_losses, lat_loss, pose_loss = extract(self.loss_idt), extract(self.loss_lat), extract(self.loss_pose)
        kl_losses = extract(self.loss_kl)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('KL', kl_losses), ('Cyc', cyc_losses), ('Idt', idt_losses), ('Sia', lat_loss), ('Pose', pose_loss)])

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.cyc_A, self.idt_A, self.mask_A, self.real_B, self.fake_A, self.rec_B, self.cyc_B, self.idt_B, self.mask_B]
            baseimgs = [self.back_A, self.back_A, self.back_A, self.back_A, self.back_A, self.back_A, self.back_B, self.back_B, self.back_B, self.back_B, self.back_B, self.back_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'cyc_A', 'idt_A', 'mask_A', 'real_B', 'fake_A', 'rec_B', 'cyc_B', 'idt_B', 'mask_B']
            sizetupple = [(self.ay,self.ax)] *6 + [(self.by,self.bx)] *6
        else:
            sizetupple = [(self.ay,self.ax)] * len(self.visuals)
            baseimgs = [self.real_A] * len(self.visuals)
        #images = [util.tensor2im(v.data) for v in self.visuals]
        images = [util.tensor2im(v.data,b.data) for v, b in zip(self.visuals,baseimgs)]
        images = [cv2.resize(img,size,interpolation=cv2.INTER_NEAREST) for img, size in zip(images,sizetupple)]
        return OrderedDict(zip(self.labels, images)), util.onten(self.rec_A), util.onten(self.rec_B)

    def save(self, label=0):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            #print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0 and self.opt.niter >= curr_iter:
            decay_frac = curr_iter / self.opt.niter
            self.lambda_enc = self.opt.lambda_latent * decay_frac
