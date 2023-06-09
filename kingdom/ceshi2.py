import torch
from torch import nn
import pickle
relation_embedding = pickle.load(open('utils/entity_embedding.pkl','rb'))
#缩小维度并转化成tensor
xx = torch.tensor([relation_embedding])
m = nn.MaxPool1d(3, stride=3)
tt = m(xx)
yy = tt.view(-1,100)
pickle.dump(yy,open('utils/max_entity_embedding.pkl','wb'))