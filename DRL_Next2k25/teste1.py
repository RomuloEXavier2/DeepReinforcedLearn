import torch
# Exemplo de criação de um tensor na GPU
if torch.cuda.is_available():
    tensor_gpu = torch.randn(10, 10).cuda()
    print(f"Tensor na GPU: {tensor_gpu.device}")
else:
    print("CUDA não disponível, tensor na CPU.")