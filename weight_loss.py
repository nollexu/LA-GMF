import torch
import torch.nn as nn

# Select logits at the patch level corresponding to the given target
def class_select(logits, target):
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    # 450*1
    return logits.masked_select(one_hot_mask)


class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()

    # alpha:3*150
    # logits:450*2
    # target:450
    # batch_size:3
    def forward(self, alpha, logits, target):
        
        softmax_logits = torch.softmax(logits, dim=1)
        print('softmax_logits', softmax_logits.shape)
        # selected_logits:450*1->3*150
        selected_logits = class_select(softmax_logits, target).view(alpha.shape[0], -1)
        print('selected_logits',selected_logits.shape)
        logit_loss = torch.norm(input=alpha - selected_logits, p=2, dim=-1).sum()
        print('logit_loss',logit_loss)

        return logit_loss
