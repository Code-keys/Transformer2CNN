...

class CSP-Conv3x3(nn.Module): 
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e) 
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential( *[ Conv( c1, c2 , 3, s=1, p=1 ,g=g) for _ in range(layer)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2) 
       
       #  : double(more)-head  
        self.m_reg = nn.ModuleList( nn.Conv2d( x,   5    * self.na, 1)  for x in ch  )   #  for reg    
        
        #  cls-inner split
        self.nc1 = 2  # split cls-mix : such as 2
        self.nc2 = nc-2  # split cls-mix
        self.cv1_cls = nn.ModuleList( CSP-Conv3x3( x, x, 3)  for x in ch  )       
        self.cv2_cls = nn.ModuleList( CSP-Conv3x3( x, x , 3) for x in ch)
        self.m1_cls = nn.ModuleList( nn.Conv2d( x, self.nc1* self.na, 1)  for x in ch )
        self.m2_cls = nn.ModuleList( nn.Conv2d( x, self.nc2* self.na, 1)  for x in ch )
 
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl): 
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            reg = self.m_reg[i](x[i]) 
            reg = reg.view(bs, self.na, 5, ny, nx).permute(0, 1, 3, 4, 2) 
   
            cls1 = self.cv1_cls[i](x[i]) 
            cls1 = self.m1_cls[i](cls1)
            cls1 = cls1.view(bs, self.na, self.nc1, ny, nx).permute(0, 1, 3, 4, 2) 
            
            cls2 = self.cv2_cls[i](x[i])
            cls2 = self.m2_cls[i](cls2)
            cls2 = cls2.view(bs, self.na, self.nc - self.nc1, ny, nx).permute(0, 1, 3, 4, 2)

            x[i] = torch.cat([reg, cls1, cls2],-1).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

...
        
