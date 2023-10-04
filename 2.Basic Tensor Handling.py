# # Part 2 Basic Tensor Handling

# ## Unsqueeze(dim)

# unsqueeze함수는 squeeze함수의 반대로 1인 차원을 생성하는 함수이다. 
# 
# 그래서 어느 차원에 1인 차원을 생성할 지 꼭 지정해주어야한다.

# In[60]:


weights = torch.tensor([0.2126, 0.7152, 0.0722])


# In[61]:


weights.shape


# In[65]:


unsqueezed_weights = weights.unsqueeze(dim=-1)


# In[67]:


unsqueezed_weights.shape


# In[68]:


unsqueezed_weights = weights.unsqueeze(dim=-1).unsqueeze(dim=-1)


# In[69]:


unsqueezed_weights.shape


# ## Transpose

# In[70]:


a = torch.ones(3, 2)
a_t = a.transpose(0, 1)


# In[71]:


a.shape, a_t.shape


# 파이토치 온라인 문서(http://pytorch.org/docs)에 텐서 연산이 완벽하게 잘 정리되어 있다. 
# 
# 
# 해당 문서는 다음과 같이 구성되어 있다.
# 
# 
# ● Creation ops: ones나 from_numpy같이 텐서를 만드는 함수
# 
# 
# ● Indexing, slicing, joining, mutating ops: 셰이프shape, 스트라이드stride, transpose처럼 텐서 
# 내부를 바꾸는 함수
# 
# 
# ● Math ops: 연산을 통해 텐서 내부를 조작하는 함수
# 
# 
# - Pointwise ops: abs나 cos처럼 텐서 요소 하나 하나에 대한 함수 실행 결과로 새 텐서를 
# 만드는 함수
# 
# - Reduction ops: mean, std, norm처럼 여러 텐서를 순회하며 집계하는 함수
# 
# - Comparison ops: equal이나 max처럼 텐서 요소에 대해 참거짓을 숫자로 평가해서 반환
# 하는 함수
# 
# - Spectral ops: stft나 hamming_window처럼 주파수 영역에 대해 변환이나 연산을 수행하
# 는 함수
# 
# - Other operations: cross처럼 벡터에 대한 연산을 수행하거나, trace처럼 행렬에 대한 특
# 수 연산을 실행하는 함수
# 
# - BLAS and LAPACK operations: 기본 선형 대수 서브프로그램BLAS, Basic Linear Algebra 
# Subprogram 정의를 따르며 스칼라, 벡터-벡터, 행렬-벡터, 행렬-행렬에 대해 연산을 수
# 행할 수 있는 함수
# 
# 
# ● Random sampling: randn이나 normal처럼 확률 분포에 기반해서 난수를 만드는 함수
# 
# 
# ● Serialization: load와 save처럼 텐서를 저장하거나 불러오는 함수
# 
# 
# ● Parallelism: set_num_threads처럼 병렬 CPU 처리 시 스레드 수를 제어하는 함수
# 
# 

# ## _ : 텐서 내부 연산

# 밑줄(_)로 끝나는데 연산의 결과로 새 텐서가 넘어오는 대신 기존 텐서의 내용이 바뀐다. 
# 
# 예를 들어 zero_ 메소드는 입력된 텐서의 모든 요소를 0으로 바꾼다. 

# In[72]:


a = torch.ones(3, 2)
a.zero_()


# ## Tensor를 GPU로 옮기기

# In[75]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# points_gpu = points.to(device='cuda')


# ## Numpy ⬅️ ➡️ Tensor

# In[77]:


# Tensor to Numpy
points = torch.ones(3, 4)
points_np = points.numpy()
points_np


# In[79]:


# Numpy to Tensor
points = torch.from_numpy(points_np)
points


# ## Tensor Serialization

# 텐서 안에 담긴 데이터가 중요하다면 이를 파일에 보관했다가 나중에 다시 읽어들이고 싶을 것이다.
# 
# 누구도 프로그램을 실행할 때마다 모델을 처음부터 재훈련시키를 원하지는 않을 것이다. 
# 
# 파이토치는 텐서 객체를 직렬화하기 위해 내부적으로 pickle을 사용하며, 저장 공간을 위한 전용 직렬화 코드를 가지고 있다. 
# 
# 다음은 우리의 points 텐서를 ourpoints.t 파일에 저장하는 코드다.

# In[85]:


points = torch.ones(3, 4)
# save
torch.save(points, './Pytorch/data/ourpoints.t')


# In[86]:


# load
points = torch.load('./Pytorch/data/ourpoints.t')


# In[87]:


points


# In[ ]:




