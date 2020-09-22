
import tadasets

from data_generation import TsneDataPair 

data = tadasets.sphere(n=1000, r=10)

sphere_tsne_dp = TsneDataPair(data)


print(sphere_tsne_dp.raw)
print(sphere_tsne_dp.embedded)
print(sphere_tsne_dp.raw.shape)
print(sphere_tsne_dp.embedded.shape)

