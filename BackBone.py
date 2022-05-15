...
class DConv(nn.Module):
    # Standard DConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=2, d=2, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding= p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

... 

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.cv3 = DConv(c_, c2, k=3, s=1, p=2, d=2, g=g)
        self.att = CBAM() 
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 =  self.att( self.cv3( self.cv2( self.cv1( x))))
        if self.add : 
            return x + x1
        else :
            return x1
