import cv2
import numpy as np
import torch 
from torch.nn import Module

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA) and
    Exponential Moving Average (EMA).
    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).
    Exponential Moving Average is a variation of `Polyak averaging`_,
    but using exponential weights instead of equal weights across iterations.
    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.
    Args:
        model (torch.nn.Module): model to use with SWA/EMA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        multi_avg_fn (function, optional): the averaging function used to update
            parameters inplace; the function must take in the current values of the
            :class:`AveragedModel` parameters as a list, the current values of :attr:`model`
            parameters as a list, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)
    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)
    You can also use custom averaging functions with the `avg_fn` or `multi_avg_fn` parameters.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights (SWA).
    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_model = torch.optim.swa_utils.AveragedModel(model,
        >>>             torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)
    .. note::
        When using SWA/EMA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.
    .. note::
        :attr:`avg_fn` and `multi_avg_fn` are not saved in the :meth:`state_dict` of the model.
    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.
    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    .. _Polyak averaging:
        https://paperswithcode.com/method/polyak-averaging
    """
    def __init__(self, model, device=None, avg_fn=None, multi_avg_fn=None, use_buffers=False):
        super().__init__()
        assert avg_fn is None or multi_avg_fn is None, 'Only one of avg_fn and multi_avg_fn should be provided'
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        self_param_detached = []
        model_param_detached = []
        for p_averaged, p_model in zip(self_param, model_param):
            device = p_averaged.device
            p_model_ = p_model.detach().to(device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if self.n_averaged == 0:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped_tensors = _group_tensors_by_device_and_dtype([self_param_detached, model_param_detached])
                for ((device, _), [self_params, model_params]) in grouped_tensors.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                    elif device.type == 'cuda':
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        for p_averaged, p_model in zip(self_params, model_params):
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
            else:
                for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    p_averaged.detach().copy_(self.avg_fn(p_averaged.detach(), p_model, n_averaged))

        if not self.use_buffers:
            # If not apply running averages to the buffers,
            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1


def img_padding(img, desired_size=260):
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    #plt.imshow(new_im),plt.show()
    return new_im
