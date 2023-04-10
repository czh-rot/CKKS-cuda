#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include "word128.h"

#define random(x) (rand()%x)
#define threadsPerBlock 256
#define word64 unsigned long long

const int m = pow(2, 2); // degree
const int n = 45; // L+1
const int r = pow(2, 1); // random number, unused


const int log_degree=14;//16
const int level=44;
const int dnum=3;
const word64 primes[]={
                    2305843009218281473, 2251799661248513,    2251799661641729,
                    2251799665180673,    2251799682088961,    2251799678943233,
                    2251799717609473,    2251799710138369,    2251799708827649,
                    2251799707385857,    2251799713677313,    2251799712366593,
                    2251799716691969,    2251799714856961,    2251799726522369,
                    2251799726129153,    2251799747493889,    2251799741857793,
                    2251799740416001,    2251799746707457,    2251799756013569,
                    2251799775805441,    2251799763091457,    2251799767154689,
                    2251799765975041,    2251799770562561,    2251799769776129,
                    2251799772266497,    2251799775281153,    2251799774887937,
                    2251799797432321,    2251799787995137,    2251799787601921,
                    2251799791403009,    2251799789568001,    2251799795466241,
                    2251799807131649,    2251799806345217,    2251799805165569,
                    2251799813554177,    2251799809884161,    2251799810670593,
                    2251799818928129,    2251799816568833,    2251799815520257,
                    2305843009218936833, 2305843009220116481, 2305843009221820417,
                    2305843009224179713, 2305843009225228289, 2305843009227980801,
                    2305843009229160449, 2305843009229946881, 2305843009231650817,
                    2305843009235189761, 2305843009240301569, 2305843009242923009,
                    2305843009244889089, 2305843009245413377, 2305843009247641601,
};
int degree;
int chain_length;
int max_num_moduli;
int alpha_;

__inline__ __device__ void mult_64_64_128(word64 x, word64 y, uint128 &ans)
{
    uint64_t a0 = (uint32_t)(x);
    uint64_t a1 = (uint32_t)(x>> 0x20);
    uint64_t b0 = (uint32_t)(y);
    uint64_t b1 = (uint32_t)(y>> 0x20);
    ans<<= 0x20;
    ans+= a1*b1;
    ans<<= 0x20;
    ans+= a1*b0;
    ans+= a0*b1;
    ans<<= 0x20;
    ans+= a0*b0;
}

__inline__ __device__ word64
barrett_reduction_128_64(const uint128 in, const word64 prime,
                        const word64 barret_ratio, const word64 barret_k) {
  uint128 temp1;
  uint128 temp2;
  mult_64_64_128(in.lo, barret_ratio, temp1);
  mult_64_64_128(in.hi, barret_ratio, temp2);
  // carry = add_64_64_carry(temp1.hi, temp2.lo, temp1.hi);
  word64 old=temp1.hi;
  temp1.hi+=temp2.lo;
  temp2.hi+=(old<temp1.hi);
  //asm("add.cc.u64 %0, %0, %1;" : "+l"(temp1.hi) : "l"(temp2.lo));
  // carry = add_64_64_carry(temp2.hi, 0, temp2.hi, carry);
  //asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(temp2.hi) : "l"((unsigned long)0));
  temp1.hi >>= barret_k - 64;
  temp2.hi <<= 128 - barret_k;
  temp1.hi = temp1.hi + temp2.hi;
  temp1.hi = temp1.hi * prime;
  word64 res = in.lo - temp1.hi;
  //printf("***%lld %lld %lld\n",res,in.hi,in.lo);
  if (res >= prime) res -= prime;
  return res;
}


__global__ void gpu_hadd(   
                            
                            word64* c1_a, word64* c1_b, 
                            word64* c2_a, word64* c2_b, 
                            int log_degree,
                            int degree, int length,
                            word64* gprimes,
                            word64* d0, word64* d1
                            ) 
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //if(threadIdx.x==0)
    //    printf("%d %d\n",degree, length); 
    //printf("Call\n");
    //for(int tid=threadIdx.x + blockIdx.x * blockDim.x; tid< degree * length ; tid+=blockDim.x * gridDim.x)
    int now=tid;
    word64 res1, res2;
    for(int i=0; i<length; i++)
    {
        auto prime=gprimes[i];
        res1=c1_a[now]+c2_a[now];
        res2=c1_b[now]+c2_b[now];
        // if(res1<c1_a[tid]&&res1<c2_a[tid])
        //     res1-=prime;
        // if(res2<c1_b[tid]&&res2<c2_b[tid])
        //     res2-=prime;
        if(prime-res1>>63)res1-=prime;
        if(prime-res2>>63)res2-=prime;
        //if(tid>=degree*length)
        //    printf("%d\n",tid);
        d0[now]=res1;
        d1[now]=res2;
        now+=degree;
        //printf("%d %d\n",tid,d0[tid]);
        //printf("%d ",tid);
    }
}

//simulate computer rand process
static word64 seed=19260817;
word64 randword64()
{
    return seed=seed*20221115+19;
}

using namespace std; 

void init()
{
    degree=1<<log_degree;
    chain_length=level+1;
    alpha_=(level+1)/dnum;
    max_num_moduli=level+1+alpha_;
}

int main(int argc, char const* argv[])
{
    //初始化
    init();

    word64 *c1_a, *c1_b;
    word64 *c2_a, *c2_b;
    word64 *d0;
    word64 *d1;
    int sumlength=degree * (level+1);
    c1_a = (word64*)malloc(sizeof(word64) * sumlength);
    c1_b = (word64*)malloc(sizeof(word64) * sumlength);
    c2_a = (word64*)malloc(sizeof(word64) * sumlength);
    c2_b = (word64*)malloc(sizeof(word64) * sumlength);
    d0 = (word64*)malloc(sizeof(word64) * sumlength);
    d1 = (word64*)malloc(sizeof(word64) * sumlength);

    //get rand polynomials
    int now=0;
    for (auto i = 0; i < level; i++) {
        auto prime=primes[i];
        for(auto j = 0; j< degree; j++){
            c1_a[now] = randword64()%prime;
            c1_b[now] = randword64()%prime;
            c2_a[now] = randword64()%prime;
            c2_b[now] = randword64()%prime;
            d0[now]=0;
            d1[now]=0;
            //cout<<i<<" "<<j<<" "<<c1_a[now]<<" "<<c1_b[now]<<" "<<c2_a[now]<<" "<<c2_b[now]<<endl;
            now++;
        }
        
    }

    word64 *gc1_a, *gc1_b;
    word64 *gc2_a, *gc2_b;
    cudaMalloc(&gc1_a, sizeof(word64) * sumlength);
    cudaMalloc(&gc1_b, sizeof(word64) * sumlength);
    cudaMalloc(&gc2_a, sizeof(word64) * sumlength);
    cudaMalloc(&gc2_b, sizeof(word64) * sumlength);

    word64 *gd0;
    word64 *gd1;
    cudaMalloc(&gd0, sizeof(word64) * sumlength);
    cudaMalloc(&gd1, sizeof(word64) * sumlength);

    word64 *gprimes;
    cudaMalloc(&gprimes, sizeof(word64) * max_num_moduli);

    // copy polynomial from host to device memory
    cudaMemcpy(gc1_a, c1_a, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gc1_b, c1_b, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gc2_a, c2_a, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gc2_b, c2_b, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gprimes, primes, sizeof(word64) * max_num_moduli, cudaMemcpyHostToDevice);
    cudaMemcpy(gd0, d0, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gd1, d1, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);

    float time_elapsed;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //HAdd
    printf("%d\n",sumlength);
    printf("%d %d\n",degree,level);
    const int gridDim = 256;
    const int blockDim = 256;
    gpu_hadd<<<128, 128>>>( gc1_a, gc1_b, 
                            gc2_a, gc2_b, 
                            log_degree, degree, level, 
                            gprimes,
                            gd0, gd1);
    //gpu_h_add<<<1024,256>>>();
    //cudaMemcpy(d0, gd0, sizeof(int) * sumlength, cudaMemcpyDeviceToHost);
    //cudaMemcpy(d1, gd1, sizeof(int) * sumlength, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("time:%fms\n",time_elapsed);

    //cout<<"?????"<<endl;
    //printf("%ld\n",sizeof(ans_a));
    // for (int i = 0; i < degree; ++i) {
    //     printf("%lld\t",d0[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < sumlength; ++i) {
    //     printf("%lld\t",d1[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < sumlength; ++i) {
    //     printf("%lld\t",d2[i]);
    // }
    // printf("\n");
    // cudaFree(gc1_a);
    // cudaFree(gc1_b);
    // cudaFree(gc2_a);
    // cudaFree(gc2_b);
    // cudaFree(&gprimes);
    // cudaFree(&gdegree);
    // cudaFree(&glength);
    // cudaFree(&glog_degree);

    // cudaFreeHost(c1_a);
    // cudaFreeHost(c1_b);
    // cudaFreeHost(c2_a);
    // cudaFreeHost(c2_b);
    // cudaFreeHost(d0);
    // cudaFreeHost(d1);
    return 0;
}

