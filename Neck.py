...

class T2Conv_s2(nn.Module): 
    # def __init__(self, ch, ch2, sam_kernel=3, s=2, ratio=16):
    def __init__(self, ch, ch2, sam_kernel=5, s=2, ratio=16):
        super().__init__()

        self.sam = Spartial_Attention( sam_kernel ) 
        self.refine = Contract()
        # self.refine = nn.UpsamplingNearest2d(2)
        self.cem = Channel_Attention( 4 * ch, ch2 ,ratio )

    def forward(self, xs=[] ):
        upper, cur = xs[0], xs[1] 
        upper = self.refine(upper)  
        hotpot = self.sam(cur) 
        # hotpot = self.refine(hotpot)
        upper = hotpot * upper  
        upper = self.cem(upper)

        return upper 
      
...

class Channel_Attention(nn.Module):
    def __init__(self, channel1, channel2, r=16):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential( 
            nn.Conv2d(channel1, channel1//r, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel1//r, channel1, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()
        self.cv1 = Conv(channel1, channel2, 1) 
    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1) 
        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return self.cv1(x * y)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        ) 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Spartial_Attention(nn.Module):
    def __init__(self, kernel_size=5):
        super(Spartial_Attention, self).__init__() 
        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2 
        self.__layer = nn.Sequential(
            # nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(2, 2, kernel_size=3, groups=2, padding=1),
            nn.Conv2d(2, 1, kernel_size=3, padding=2, dilation=2), 
            nn.Sigmoid(),
        ) 
    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1) 
        mask = self.__layer(mask) 
        return  mask 

