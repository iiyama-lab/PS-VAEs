import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def define_G(input_nc, output_nc, ngf, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[], out_classes = 32, divide_dims=192,oldmodel=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, divide_dims, norm_layer, use_dropout, gpu_ids, use_bias, oldmodel)
    enc_args = [(input_nc, n_blocks_enc) + dup_args, (output_nc, n_blocks_enc) + dup_args]
    dec_args = [(input_nc, n_blocks_dec) + dup_args, (output_nc, n_blocks_dec) + dup_args]

    if n_blocks_shared > 0:
        n_blocks_shdec = n_blocks_shared // 2
        n_blocks_shenc = n_blocks_shared - n_blocks_shdec
        shenc_args = (n_domains, n_blocks_shenc) + dup_args
        shdec_args = (n_domains, n_blocks_shdec) + dup_args
        plex_netG = G_Plexer(n_domains, out_classes, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, LabelEncoder, LabelMaker, ResnetGenShared, shenc_args, shdec_args)
    else:
        plex_netG = G_Plexer(n_domains, out_classes, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, LabelEncoder, LabelMaker)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, output_nc, ndf, netD_n_layers, n_domains, blur_fn, norm='batch', s_norm = False, gpu_ids=[], num_feature_nc = 256):
    norm_layer = get_norm_layer(norm_type=norm)

    #model_args = [(input_nc, ndf, netD_n_layers, blur_fn, norm_layer, gpu_ids), (output_nc, ndf, netD_n_layers, blur_fn, norm_layer, gpu_ids), (num_feature_nc, ndf, netD_n_layers, blur_fn, norm_layer, gpu_ids)]
    model_args = [(input_nc, ndf, netD_n_layers, blur_fn, norm_layer, s_norm, gpu_ids), (output_nc, ndf, netD_n_layers, blur_fn, norm_layer, s_norm, gpu_ids)]

    if n_domains == 3:
        #CycADA
        model_args.append((num_feature_nc, ndf, netD_n_layers, blur_fn, norm_layer, s_norm, gpu_ids))

    plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, tensor=torch.FloatTensor, flatloss=False):
        super(GANLoss, self).__init__()
        self.Tensor = tensor
        #self.labels_real, self.labels_fake = None, None
        self.preloss = nn.Sigmoid() if not use_lsgan else None
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()
        self.flatloss = flatloss

    def get_target_tensor(self, inputs, is_real):
        #if self.labels_real is None or self.labels_real[0].numel() != inputs[0].numel():
        #   self.labels_real = [ self.Tensor(input.size()).fill_(1.0) for input in inputs ]
        #    self.labels_fake = [ self.Tensor(input.size()).fill_(0.0) for input in inputs ]
        labels_real = [ self.Tensor(input.size()).fill_(1.0) for input in inputs ]
        labels_fake = [ self.Tensor(input.size()).fill_(-1.0) for input in inputs ]

        if is_real:
            return labels_real
        return labels_fake

    def __call__(self, inputs, is_real):
        labels = self.get_target_tensor(inputs, is_real)
        #print([x.size() for x in inputs])
        #print([x.size() for x in labels])
        if self.preloss is not None:
            inputs = [self.preloss(input) for input in inputs]
        losses = [self.loss(input, label) for input, label in zip(inputs, labels)]
        multipliers = list(range(1, len(inputs)+1));  multipliers[-1] += 1
        if self.flatloss:
            multipliers = [1] * len(inputs)
        losses = [m*l for m,l in zip(multipliers, losses)]
        return sum(losses) / (sum(multipliers) * len(losses))

class Upsampling(nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling, self).__init__()
        self.interp = F.upsample_nearest
        #self.size = size
        self.scale_factor = scale_factor
        #self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor)
        return x

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, divide_dims = 192, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, old=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids
        ngf = ngf // 4
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)
        self.mumodel = nn.Sequential(*[nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1, bias=use_bias)])
        self.logvarmodel = nn.Sequential(*[nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1, bias=use_bias),nn.Softplus()])


    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return nn.parallel.data_parallel(self.mumodel, x, self.gpu_ids), nn.parallel.data_parallel(self.logvarmodel, x, self.gpu_ids)
        x = self.model(input)
        return self.mumodel(x), self.logvarmodel(x)

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, divide_dims=192, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)


class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, divide_dims=192, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, old=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids
        ngf = ngf // 4
        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
        model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1)]
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [Upsampling(scale_factor=2),
                       nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3,
                                         padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class LabelEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, divide_dims=192, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False,old=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(LabelEncoder, self).__init__()
        self.gpu_ids = gpu_ids
        ngf = divide_dims // 4
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LabelMaker(nn.Module):
    def __init__(self, out_classes, output_nc, n_blocks=5, ngf=64, divide_dims=192, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, old=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(LabelMaker, self).__init__()
        self.out_classes = out_classes
        self.gpu_ids = gpu_ids
        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
        ngf = divide_dims // 4

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [Upsampling(scale_factor=2),
                       nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3,
                                         padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        mnist = [nn.PReLU(),nn.Dropout2d(),Flatten(), nn.Linear(divide_dims*8*8,10)]

        if self.out_classes == 10:
            self.mnistout = nn.Sequential(*mnist)
        else:
            self.model = nn.Sequential(*model)
            self.labelout = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(ngf, 32, kernel_size=7, padding=0)])
            self.jointout = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(ngf, 18, kernel_size=7, padding=0)])

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            if self.out_classes == 10:
                return [nn.parallel.data_parallel(self.mnistout, input, self.gpu_ids)]
            x = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            if self.out_classes == 18:
                return [nn.parallel.data_parallel(self.jointout, x, self.gpu_ids)]
            elif self.out_classes == 32:
                return [nn.parallel.data_parallel(self.labelout, x, self.gpu_ids)]
            elif self.out_classes == 50:
                return [nn.parallel.data_parallel(self.jointout, x, self.gpu_ids), nn.parallel.data_parallel(self.labelout, x, self.gpu_ids)]
        else:
            if self.out_classes == 10:
                return [self.mnistout(input)]
            x = self.model(input)
            if self.out_classes == 18:
                return [self.jointout(x)]
            elif self.out_classes == 32:
                return [self.labelout(x)]
            elif self.out_classes == 50:
                return [self.jointout(self.model(input)), self.labelout(self.model(input))]

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()
#ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        #self.conv_block = nn.Sequential(*conv_block)
        self.conv_block = SequentialContext(n_domains, *conv_block)
    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, blur_fn=None, norm_layer=nn.BatchNorm2d, s_norm = False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.blur_fn = blur_fn
        self.s_norm = s_norm
        self.gray_fn = lambda x: (.2126*x[:,0,:,:] + .7152*x[:,1,:,:] + .0722*x[:,2,:,:])[:,None,:,:]
        #self.gray_fn = lambda x: (x[:,0,:,:])[:,None,:,:]
        fea_model = []
        if n_layers == 0:
            m = [Flatten(), nn.Linear(input_nc*8*8,1024),nn.PReLU(), nn.Linear(1024,256),nn.PReLU(), nn.Linear(256,1)]
            self.model_rgb = nn.Sequential(*m)
        else:
            self.model_gray = self.model(1, ndf, n_layers, norm_layer)
            self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        #self.input_nc = input_nc
    def use_s_norm(self, inp, outp, k, s, p, b = None):
        if self.s_norm:
            if b is None:
                return spectral_norm(nn.Conv2d(inp, outp, kernel_size=k, stride=s, padding=p))
            else:
                return spectral_norm(nn.Conv2d(inp, outp, kernel_size=k, stride=s, padding=p, bias = b))
        else:
            if b is None:
                return nn.Conv2d(inp, outp, kernel_size=k, stride=s, padding=p)
            else:
                return nn.Conv2d(inp, outp, kernel_size=k, stride=s, padding=p, bias = b)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc > ndf:
            ndf = input_nc // 4 * 4
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            self.use_s_norm(input_nc, ndf, kw, 2, padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                self.use_s_norm(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kw, 2, padw, use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            self.use_s_norm(ndf * nf_mult_prev, ndf * nf_mult,
                      kw, 1, padw, use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input, input_nc, use_Gaussian=True):
        blurred_rgb = self.blur_fn(input,input_nc)
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs2 = nn.parallel.data_parallel(self.model_rgb, blurred_rgb, self.gpu_ids)
        else:
            outs2 = self.model_rgb(blurred_rgb)
        return outs2

class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        for opt in self.optimizers:
            opt.zero_grad()

    def step_grads(self, dom_a, dom_b):
        for opt in self.optimizers:
            opt.step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, out_classes, encoder, enc_args, decoder, dec_args, labelencoder, labelmaker,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args[i]) for i in range(n_domains)]
        self.decoders = [decoder(*dec_args[i]) for i in range(n_domains)]
        self.arg = enc_args[0][:3]
        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.labelencoder = labelencoder(*enc_args[0])
        self.labelmaker = labelmaker(out_classes, *dec_args[0])
        self.encoders.append( self.labelencoder )
        self.decoders.append( self.labelmaker )

        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def forward(self, input, in_domain=0, out_domain=0):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def labelencode(self, input, domain=0):
        return self.labelencoder.forward(input)

    def labeldecode(self, input, domain=0):
        return self.labelmaker.forward(input)

    def encode(self, input, domain=0):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain=0):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        #self.optimizers[0].zero_grad()

        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-2].zero_grad()
        self.optimizers[dom_b].zero_grad()
        self.optimizers[-1].zero_grad()

    def step_grads(self, dom_a, dom_b):
        #self.optimizers[0].step()

        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-2].step()
        self.optimizers[dom_b].step()
        self.optimizers[-1].step()

    def __repr__(self):
        e, d, l, m = self.encoders[0], self.decoders[0], self.labelencoder, self.labelmaker
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        l_params = sum([p.numel() for p in l.parameters()])
        m_params = sum([p.numel() for p in m.parameters()])

        return repr(e) +'\n'+ repr(d) +'\n'+ repr(l) +'\n'+ repr(m) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Decoder: %d' % d_params +'\n'+ \
            'Number of parameters per LabelEncoder: %d' % l_params +'\n'+ \
            'Number of parameters per LabelMaker: %d' % m_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args[i]) for i in range(n_domains)]
        self.arg = model_args[0][:3]
    def forward(self, input, domain):
        discriminator = self.networks[domain]
        input_nc = input.shape[1]
        return discriminator.forward(input, input_nc)
    """
    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()
    """
    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])

        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes # 0
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
