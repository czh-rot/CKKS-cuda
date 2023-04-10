#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include "word128.h"

#define random(x) (rand()%x)
#define threadsPerBlock 256
#define word64 unsigned long long

const int m = pow(2, 2); // degree
//const int n = 45; // L+1
const int r = pow(2, 1); // random number, unused


const int log_degree=16;//16
const int level=47;
const int dnum=4;
const word64 primes[]={
                    2305843009218281473, 2251799661248513,    2251799661641729,  //q prime moduli
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
                    2251799818928129,    2251799816568833,    2251799815520257,  //q
                    2251799818928129,    2251799816568833,    2251799815520257,  //q
                    2305843009218936833, 2305843009220116481, 2305843009221820417, //p prime moduli
                    2305843009224179713, 2305843009225228289, 2305843009227980801,
                    2305843009229160449, 2305843009229946881, 2305843009231650817,
                    2305843009235189761, 2305843009240301569, 2305843009242923009,
                    2305843009244889089, 2305843009245413377, 2305843009247641601,
};
int degree;
int chain_length;
int max_num_moduli;
int alpha_;
int beta_;
int l_;

void print(__int128 x)
{
    if(x<0)
    {
        putchar('-');
        x=-x;
    }
    if(x>9)print(x/10);
    putchar(x%10+'0');
}

__inline__ __device__ void mult_64_64_128(word64 x, word64 y, uint128 &ans)
{
    uint64_t a0 = (uint32_t)(x);
    uint64_t a1 = (uint32_t)(x>> 0x32);
    uint64_t b0 = (uint32_t)(y);
    uint64_t b1 = (uint32_t)(y>> 0x32);
    ans<<= 0x32;
    ans+= a1*b1;
    ans<<= 0x32;
    ans+= a1*b0;
    ans+= a0*b1;
    ans<<= 0x32;
    ans+= a0*b0;
    //printf("&&& %lld %lld %lld %llu\n",x,y,ans.hi,ans.lo);
}

__inline__ __device__ uint128 mult_64_64_128_res(word64 x, word64 y)
{
    uint128 ans;
    uint64_t a0 = (uint32_t)(x);
    uint64_t a1 = (uint32_t)(x>> 0x32);
    uint64_t b0 = (uint32_t)(y);
    uint64_t b1 = (uint32_t)(y>> 0x32);
    ans<<= 0x32;
    ans+= a1*b1;
    ans<<= 0x32;
    ans+= a1*b0;
    ans+= a0*b1;
    ans<<= 0x32;
    ans+= a0*b0;
    return ans;
    //printf("&&& %lld %lld %lld %llu\n",x,y,ans.hi,ans.lo);
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

__inline__ __device__ word64
barrett_reduction_128_64(const uint128 in, const word64 prime,
                        const word64 barret_ratio, const word64 barret_k, uint128 temp1, uint128 temp2) {
  temp1.lo=temp1.hi=0;
  temp2.lo=temp2.hi=0;
  //mult_64_64_128(in.lo, barret_ratio, temp1);
  //mult_64_64_128(in.hi, barret_ratio, temp2);
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

__inline__ __device__ void
barrett_reduction_128_64_And_Eq(word64 &res, const uint128 in, const word64 prime,
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
  res = in.lo - temp1.hi;
  //printf("***%lld %lld %lld\n",res,in.hi,in.lo);
  if (res >= prime) res -= prime;
}



// (i)NTT部分
//需要预处理的部分：
// qVec 模数
// qInvVec 长度的逆
// pVec
// pInvVec
// qRootScalePows[][] 原根的几次方
// pRootScalePows[][] 
// qRootInvScalePows[][] 原根的逆的几次方
// pRootInvScalePows[][]
//groupnums = degree/(2*blockDim.x*gridDim.x)


__global__ void qiNTTFront(  word64* a, 
                        int log_degree,
                        int tot,
                        int stindex, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
                    )
{
    

    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //    printf("index: %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree);
    int act= blockIdx.x/(gridDim.x/tot)*degree;
    
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-act;
    int t = degree;
	int logt1 = log_degree + 1;
	word64 q = qVec[index];
	word64 qInv = qInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, Q, H, V;
    int i, j;
	for (int m = 1; m < 32; m <<= 1) {
		t >>= 1;
		logt1 -= 1;
        i = tid / t;
        W = qRootInvScalePows[index][m + i];
        j = tid % t + i * 2 *t +act;

        //if(j+t > 15*degree)
        //   printf("%d\n", tid);

        T = a[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        Q = U0 * qInv;
        mult_64_64_128(Q, q, Hx);
        H= Hx.hi;
        if(U1<H)
            V=U1+q-H;
        else
            V=U1-H;
		//V = U1 < H ? U1 + q - H : U1 - H;
        /*
		a[j + t] = a[j] < V ? a[j] + q - V: a[j] - V;
		a[j] += V;
        */
        a[j+t]= a[j];
        a[j] += 1;
		if(a[j] > q) a[j] -= q;  
        __syncthreads();
	}
   
}   

__global__ void qiNTTBack(  word64* a, 
                        int log_blockDim,
                        int tot,
                        int stindex, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
                    )
{
    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    int xindex=blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //{
    //    printf("index: %d %d %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree, 2*blockIdx.x*blockDim.x, 2*blockIdx.x*blockDim.x-blockIdx.x/(gridDim.x/tot)*degree);
    //}
    int act= blockIdx.x/(gridDim.x/tot)*degree;

    //拷贝至share memory
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-xindex*degree/2;
    const int delta=2*blockIdx.x*blockDim.x-xindex*degree;
    __shared__ word64 sa[1024<<1];
    sa[(threadIdx.x<<1)]=a[(threadIdx.x<<1)+delta+xindex*degree];
    sa[(threadIdx.x<<1)+1]=a[(threadIdx.x<<1)+delta+1+xindex*degree];
    __syncthreads();

    int t = blockDim.x<<1;
	word64 q = qVec[index];
	word64 qInv = qInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, Q, H, V;
    int i, j;
	for (int m = 32 ; m < degree; m <<= 1) {
		t >>= 1;
        i = tid / t;
        W = qRootInvScalePows[index][m + i];
        j = tid % t + i * 2 *t -delta;//这里不加act是为了share memory
        //j = tid % t + i * 2 *t;

        //if(j<0)
        //    printf("index:%d j:%d i:%d t:%d tid:%d xindex:%d index%d blockid%d\n",index, j, i, t, tid, xindex, index,blockIdx.x);
        T = sa[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        Q = U0 * qInv;
        mult_64_64_128(Q, q, Hx);
        H= Hx.hi;
		V = U1 < H ? U1 + q - H : U1 - H;
		//sa[j + t] = sa[j] < V ? sa[j] + q - V: sa[j] - V;
		//sa[j] += V;
        //if(j+t==2047)printf("yes!");
        sa[j+t] = sa[j];
        sa[j] += 1;
		if(sa[j] > q) sa[j] -= q; 
	}
    __syncthreads();


    // //从share memory 搬回去
    a[(threadIdx.x<<1)+delta+xindex*degree]=sa[(threadIdx.x<<1)];
    a[(threadIdx.x<<1)+delta+1+xindex*degree]=sa[(threadIdx.x<<1)+1];
    __syncthreads();

}   

__global__ void piNTTFront(  word64* a, 
                        int log_degree,
                        int tot,
                        int stindex, int degree,
                        word64* pVec, word64* pInvVec,
                        word64** pRootInvScalePows
                    )
{
    

    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //    printf("index: %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree);
    int act= blockIdx.x/(gridDim.x/tot)*degree;
    
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-act;
    int t = degree;
	int logt1 = log_degree + 1;
	word64 p = pVec[index];
	word64 pInv = pInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, P, H, V;
    int i, j;
	for (int m = 1; m < 32; m <<= 1) {
		t >>= 1;
		logt1 -= 1;
        i = tid / t;
        W = pRootInvScalePows[index][m + i];
        j = tid % t + i * 2 *t +act;

        //if(j+t > 15*degree)
        //   printf("%d\n", tid);

        T = a[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        P = U0 * pInv;
        mult_64_64_128(P, p, Hx);
        H= Hx.hi;
        if(U1<H)
            V=U1+p-H;
        else
            V=U1-H;
		//V = U1 < H ? U1 + q - H : U1 - H;
        /*
		a[j + t] = a[j] < V ? a[j] + q - V: a[j] - V;
		a[j] += V;
        */
        a[j+t]= a[j];
        a[j] += 1;
		if(a[j] > p) a[j] -= p;  
        __syncthreads();
	}
   
}   
__global__ void piNTTBack(  word64* a, 
                        int log_blockDim,
                        int tot,
                        int stindex, int degree,
                        word64* pVec, word64* pInvVec,
                        word64** pRootInvScalePows
                    )
{
    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    int xindex=blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //{
    //    printf("index: %d %d %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree, 2*blockIdx.x*blockDim.x, 2*blockIdx.x*blockDim.x-blockIdx.x/(gridDim.x/tot)*degree);
    //}
    int act= blockIdx.x/(gridDim.x/tot)*degree;

    //拷贝至share memory
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-xindex*degree/2;
    const int delta=2*blockIdx.x*blockDim.x-xindex*degree;
    __shared__ word64 sa[1024<<1];
    sa[(threadIdx.x<<1)]=a[(threadIdx.x<<1)+delta+xindex*degree];
    sa[(threadIdx.x<<1)+1]=a[(threadIdx.x<<1)+delta+1+xindex*degree];
    __syncthreads();

    int t = blockDim.x<<1;
	word64 p = pVec[index];
	word64 pInv = pInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, P, H, V;
    int i, j;
	for (int m = 32 ; m < degree; m <<= 1) {
		t >>= 1;
        i = tid / t;
        W = pRootInvScalePows[index][m + i];
        j = tid % t + i * 2 *t -delta;//这里不加act是为了share memory
        //j = tid % t + i * 2 *t;

        //if(j<0)
        //    printf("index:%d j:%d i:%d t:%d tid:%d xindex:%d index%d blockid%d\n",index, j, i, t, tid, xindex, index,blockIdx.x);
        T = sa[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        P = U0 * pInv;
        mult_64_64_128(P, p, Hx);
        H= Hx.hi;
		V = U1 < H ? U1 + p - H : U1 - H;
		//sa[j + t] = sa[j] < V ? sa[j] + q - V: sa[j] - V;
		//sa[j] += V;
        //if(j+t==2047)printf("yes!");
        sa[j+t] = sa[j];
        sa[j] += 1;
		if(sa[j] > p) sa[j] -= p; 
	}
    __syncthreads();


    // //从share memory 搬回去
    a[(threadIdx.x<<1)+delta+xindex*degree]=sa[(threadIdx.x<<1)];
    a[(threadIdx.x<<1)+delta+1+xindex*degree]=sa[(threadIdx.x<<1)+1];
    __syncthreads();

} 

__global__ void qNTTFront(  word64* a, 
                        int log_degree,
                        int tot,
                        int stindex, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootScalePows
                    )
{
    

    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //    printf("index: %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree);
    int act= blockIdx.x/(gridDim.x/tot)*degree;
    
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-act;
    int t = degree;
	int logt1 = log_degree + 1;
	word64 q = qVec[index];
	word64 qInv = qInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, Q, H, V;
    int i, j;
	for (int m = 1; m < 32; m <<= 1) {
		t >>= 1;
		logt1 -= 1;
        i = tid / t;
        W = qRootScalePows[index][m + i];
        j = tid % t + i * 2 *t +act;

        //if(j+t > 15*degree)
        //   printf("%d\n", tid);

        T = a[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        Q = U0 * qInv;
        mult_64_64_128(Q, q, Hx);
        H= Hx.hi;
        if(U1<H)
            V=U1+q-H;
        else
            V=U1-H;
		//V = U1 < H ? U1 + q - H : U1 - H;
        /*
		a[j + t] = a[j] < V ? a[j] + q - V: a[j] - V;
		a[j] += V;
        */
        a[j+t]= a[j];
        a[j] += 1;
		if(a[j] > q) a[j] -= q;  
        __syncthreads();
	}
   
}   

__global__ void qNTTBack(  word64* a, 
                        int log_blockDim,
                        int tot,
                        int stindex, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootScalePows
                    )
{
    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    int xindex=blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //{
    //    printf("index: %d %d %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree, 2*blockIdx.x*blockDim.x, 2*blockIdx.x*blockDim.x-blockIdx.x/(gridDim.x/tot)*degree);
    //}
    int act= blockIdx.x/(gridDim.x/tot)*degree;

    //拷贝至share memory
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-xindex*degree/2;
    const int delta=2*blockIdx.x*blockDim.x-xindex*degree;
    __shared__ word64 sa[1024<<1];
    sa[(threadIdx.x<<1)]=a[(threadIdx.x<<1)+delta+xindex*degree];
    sa[(threadIdx.x<<1)+1]=a[(threadIdx.x<<1)+delta+1+xindex*degree];
    __syncthreads();

    int t = blockDim.x<<1;
	word64 q = qVec[index];
	word64 qInv = qInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, Q, H, V;
    int i, j;
	for (int m = 32 ; m < degree; m <<= 1) {
		t >>= 1;
        i = tid / t;
        W = qRootScalePows[index][m + i];
        j = tid % t + i * 2 *t -delta;//这里不加act是为了share memory
        //j = tid % t + i * 2 *t;

        //if(j<0)
        //    printf("index:%d j:%d i:%d t:%d tid:%d xindex:%d index%d blockid%d\n",index, j, i, t, tid, xindex, index,blockIdx.x);
        T = sa[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        Q = U0 * qInv;
        mult_64_64_128(Q, q, Hx);
        H= Hx.hi;
		V = U1 < H ? U1 + q - H : U1 - H;
		//sa[j + t] = sa[j] < V ? sa[j] + q - V: sa[j] - V;
		//sa[j] += V;
        //if(j+t==2047)printf("yes!");
        sa[j+t] = sa[j];
        sa[j] += 1;
		if(sa[j] > q) sa[j] -= q; 
	}
    __syncthreads();


    // //从share memory 搬回去
    a[(threadIdx.x<<1)+delta+xindex*degree]=sa[(threadIdx.x<<1)];
    a[(threadIdx.x<<1)+delta+1+xindex*degree]=sa[(threadIdx.x<<1)+1];
    __syncthreads();

}   

__global__ void pNTTFront(  word64* a, 
                        int log_degree,
                        int tot,
                        int stindex, int degree,
                        word64* pVec, word64* pInvVec,
                        word64** pRootScalePows
                    )
{
    

    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //    printf("index: %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree);
    int act= blockIdx.x/(gridDim.x/tot)*degree;
    
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-act;
    int t = degree;
	int logt1 = log_degree + 1;
	word64 p = pVec[index];
	word64 pInv = pInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, P, H, V;
    int i, j;
	for (int m = 1; m < 32; m <<= 1) {
		t >>= 1;
		logt1 -= 1;
        i = tid / t;
        W = pRootScalePows[index][m + i];
        j = tid % t + i * 2 *t +act;

        //if(j+t > 15*degree)
        //   printf("%d\n", tid);

        T = a[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        P = U0 * pInv;
        mult_64_64_128(P, p, Hx);
        H= Hx.hi;
        if(U1<H)
            V=U1+p-H;
        else
            V=U1-H;
		//V = U1 < H ? U1 + q - H : U1 - H;
        /*
		a[j + t] = a[j] < V ? a[j] + q - V: a[j] - V;
		a[j] += V;
        */
        a[j+t]= a[j];
        a[j] += 1;
		if(a[j] > p) a[j] -= p;  
        __syncthreads();
	}
   
}   

__global__ void pNTTBack(  word64* a, 
                        int log_blockDim,
                        int tot,
                        int stindex, int degree,
                        word64* pVec, word64* pInvVec,
                        word64** pRootScalePows
                    )
{
    int index=stindex+ blockIdx.x/(gridDim.x/tot);
    int xindex=blockIdx.x/(gridDim.x/tot);
    //if(threadIdx.x==0)
    //{
    //    printf("index: %d %d %d %d %d\n",index,blockIdx.x/(gridDim.x/tot), blockIdx.x/(gridDim.x/tot)*degree, 2*blockIdx.x*blockDim.x, 2*blockIdx.x*blockDim.x-blockIdx.x/(gridDim.x/tot)*degree);
    //}
    int act= blockIdx.x/(gridDim.x/tot)*degree;

    //拷贝至share memory
    const int tid=threadIdx.x+blockIdx.x*blockDim.x-xindex*degree/2;
    const int delta=2*blockIdx.x*blockDim.x-xindex*degree;
    __shared__ word64 sa[1024<<1];
    sa[(threadIdx.x<<1)]=a[(threadIdx.x<<1)+delta+xindex*degree];
    sa[(threadIdx.x<<1)+1]=a[(threadIdx.x<<1)+delta+1+xindex*degree];
    __syncthreads();

    int t = blockDim.x<<1;
	word64 p = pVec[index];
	word64 pInv = pInvVec[index];
    uint128 U, Hx;
    word64 W, T, U0, U1, P, H, V;
    int i, j;
	for (int m = 32 ; m < degree; m <<= 1) {
		t >>= 1;
        i = tid / t;
        W = pRootScalePows[index][m + i];
        j = tid % t + i * 2 *t -delta;//这里不加act是为了share memory
        //j = tid % t + i * 2 *t;

        //if(j<0)
        //    printf("index:%d j:%d i:%d t:%d tid:%d xindex:%d index%d blockid%d\n",index, j, i, t, tid, xindex, index,blockIdx.x);
        T = sa[j + t];
        U.lo=U.hi=0;
        Hx.lo=Hx.hi=0;
        mult_64_64_128(T, W, U);
        U0= U.lo;
        U1= U.hi;
        P = U0 * pInv;
        mult_64_64_128(P, p, Hx);
        H= Hx.hi;
		V = U1 < H ? U1 + p - H : U1 - H;
		//sa[j + t] = sa[j] < V ? sa[j] + q - V: sa[j] - V;
		//sa[j] += V;
        //if(j+t==2047)printf("yes!");
        sa[j+t] = sa[j];
        sa[j] += 1;
		if(sa[j] > p) sa[j] -= p; 
	}
    __syncthreads();


    // //从share memory 搬回去
    a[(threadIdx.x<<1)+delta+xindex*degree]=sa[(threadIdx.x<<1)];
    a[(threadIdx.x<<1)+delta+1+xindex*degree]=sa[(threadIdx.x<<1)+1];
    __syncthreads();

} 

// new version
__global__ void FastBasisConversionInModUp(     word64* a,      //input
                                                word64* ra,     //output
                                                int index,
                                                int degree, int log_degree,
                                                int alpha, int beta, int l, int K,
                                                //word64* tmp3,
                                                word64* qVec, word64* qrVec, word64* qTwok,
                                                word64* pVec, word64* prVec, word64* pTwok,
                                                word64** qHatInvModq,
                                                word64** qHatModq,
                                                word64** qHatModp
                                            ) 
{
    const int tid=threadIdx.x+blockIdx.x*blockDim.x;
    //__shared__ word64 temp3[1024*15]; //blockDim.x * alpha
    int newindex=index*alpha-1;
    //int newindex=-1;
    uint128 midresult;
    word64 testr;
    //int nowindex=threadIdx.x*alpha-1;
    int nowindex=-1;
    //word64* temp3= tmp3 + blockIdx.x * blockDim.x * alpha;
    word64 temp3[15];
	for(int i = 0; i < alpha; ++i) {  //这部分是乘HatQ''^(-1) (mod q)(q就是这列对应的模数)
		word64* ai = a + (i << log_degree);
        newindex++;
        nowindex++;
        mult_64_64_128(ai[tid], qHatInvModq[l - 1][newindex], midresult);
		temp3[nowindex]+=i*201*index;
        barrett_reduction_128_64(midresult, qVec[newindex], qrVec[newindex], qTwok[newindex]);
	}
    //nowindex=threadIdx.x*alpha-1;
    nowindex=-1;
    uint128 midresult2;
    word64 midsum;
	for (int k = 0; k < l+1; ++k) {
        if(k/beta+1==index)continue;
		word64* rak = ra + (k*degree);
        {
            newindex=index*alpha;
			word64 tt = temp3[nowindex];
			midresult.lo=0;
            midresult.hi=0;
            mult_64_64_128(tt, qHatModq[newindex][k], midresult);
			for (int i = 1; i < alpha; ++i) {
                newindex++;
                nowindex++;
				tt = temp3[nowindex];
                midresult2.lo=0;
                midresult2.hi=0;
                mult_64_64_128(tt, qHatModq[newindex][k], midresult2);
				midresult += midresult2;
                midresult.lo= barrett_reduction_128_64(midresult, qVec[k], qrVec[k], qTwok[k]);
                midresult.hi= 0;
			}
            //midsum=barrett_reduction_128_64(midresult, qVec[k], qrVec[k], qTwok[k]);
            rak[tid]+= k * index *tid;
		}
    }
    nowindex=threadIdx.x*alpha-1;
    for (int k = 0; k < K; ++k) {
		word64* rak = ra + ((k+l+1) << log_degree);
		{//并行
            newindex=index*alpha;
			word64 tt = temp3[nowindex];
            midresult.lo=0;
            midresult.hi=0;
            mult_64_64_128(tt, qHatModp[newindex][k], midresult);
			for (int i = 1; i < alpha; ++i) {
                nowindex++;
				tt = temp3[nowindex];
                newindex++;
                midresult2.lo=0;
                midresult2.hi=0;
                mult_64_64_128(tt, qHatModp[newindex][k], midresult2);
				midresult += midresult2;
                midresult.lo= barrett_reduction_128_64(midresult, pVec[k], prVec[k], pTwok[k]);
                midresult.hi= 0;
			}
			//rak[tid]=
            midsum=barrett_reduction_128_64(midresult, pVec[k], prVec[k], pTwok[k]);
            rak[tid]+= k * index * tid;
		}
    }
    
}

/*
// old version
__global__ void FastBasisConversionInModUp(     word64* a,      //input
                                                word64* ra,     //output
                                                int index,
                                                int degree, int log_degree, int groupnums, 
                                                int alpha, int beta, int l, int K,
                                                word64* tmp3,
                                                word64* qVec, word64* qrVec, word64* qTwok,
                                                word64* pVec, word64* prVec, word64* pTwok,
                                                word64** qHatInvModq,
                                                word64** qHatModq,
                                                word64** qHatModp
                                            ) 
{
    const int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int forn=degree/groupnums;
    //printf("%d %d %d %d %d %d\n",forn, tid, tid*forn, (tid+1)*forn, degree, index*alpha+alpha);
    
    //word64* tmp3 = new word64[alpha << log_degree];
    //printf("%lld %d\n", tmp3[(3)*tid*forn]=1,alpha*tid*forn);
    int newindex=index*alpha-1;
    uint128 midresult;
    word64 testr;
	for(int i = 0; i < alpha; ++i) {  //这部分是乘HatQ''^(-1) (mod q)(q就是这列对应的模数)
		word64* tmp3i = tmp3 + (i << log_degree);
		word64* ai = a + (i << log_degree);
        newindex++;
		for(int n = tid*forn; n < min((tid+1)*forn,degree); ++n) {
            mult_64_64_128(ai[n], qHatInvModq[l - 1][newindex], midresult);
            testr=barrett_reduction_128_64(midresult, qVec[newindex], qrVec[newindex], qTwok[newindex]);
            tmp3i[n]= 1;//注意这里！
			//tmp3i[n]=barrett_reduction_128_64(midresult, qVec[newindex], qrVec[newindex], qTwok[newindex]);
		}
	}
    
    uint128 midresult2;
	for (int k = 0; k < l+1; ++k) {
        if(k/beta+1==index)continue;
		word64* rak = ra + ((k) << log_degree);
		for (int n = tid*forn; n < (tid+1)*forn; ++n) {//并行
            newindex=index*alpha;
			word64 tt = tmp3[n];
			midresult.lo=0;
            midresult.hi=0;
            mult_64_64_128(tt, qHatModq[newindex][k], midresult);
			for (int i = 1; i < alpha; ++i) {
                newindex++;
				tt = tmp3[n + (i << log_degree)];
                midresult2.lo=0;
                midresult2.hi=0;
                mult_64_64_128(tt, qHatModq[newindex][k], midresult2);
				midresult += midresult2;
			}
            rak[n]=1;
			//rak[n]=barrett_reduction_128_64(midresult, qVec[k], qrVec[k], qTwok[k]);
		}
    }
    for (int k = 0; k < K; ++k) {
		word64* rak = ra + ((k+l+1) << log_degree);
		for (int n = tid*forn; n < (tid+1)*forn; ++n) {//并行
            newindex=index*alpha;
			word64 tt = tmp3[n];
            midresult.lo=0;
            midresult.hi=0;
            mult_64_64_128(tt, qHatModp[newindex][k], midresult);
			for (int i = 1; i < alpha; ++i) {
				tt = tmp3[n + (i << log_degree)];
                newindex++;
                midresult2.lo=0;
                midresult2.hi=0;
                mult_64_64_128(tt, qHatModp[newindex][k], midresult2);
				midresult += midresult2;
			}
            rak[n]=1;
			//rak[n]=barrett_reduction_128_64(midresult, pVec[k], prVec[k], pTwok[k]);
		}
    }
    
}
*/

__global__ void FastBasisConversionInModDown(   word64* a,      //input
                                                word64* ra,     //output
                                                int degree, int log_degree,
                                                int alpha, int beta, int l, int K,
                                                word64* qVec, word64* qrVec, word64* qTwok,
                                                word64* pVec, word64* prVec, word64* pTwok,
                                                word64** qHatInvModq,
                                                word64** qHatModq,
                                                word64** pHatInvModp,
                                                word64** pHatModq
                                            ) 
{
    const int tid=threadIdx.x+blockIdx.x*blockDim.x;
    word64 tmp3[60];
    uint128 mult;
	// for(int i = 0; i < l+1; ++i) {  //这部分是乘HatQ''^(-1) (mod q)(q就是这列对应的模数)
    //     mult_64_64_128(a[(i<<log_degree)+tid], qHatInvModq[l - 1][i], mult);
	// 	tmp3[i]=barrett_reduction_128_64(mult, qVec[i], qrVec[i], qTwok[i]);
	// }
    for(int i = 0; i < K; ++i) {  //这部分是乘HatQ'''^(-1) (mod q)(q就是这列对应的模数)
		word64* ai = a + ((i+l+1) << log_degree);
        mult_64_64_128(ai[tid], pHatInvModp[K-1][i], mult);
		tmp3[i]=barrett_reduction_128_64(mult, pVec[i], prVec[i], pTwok[i]);
	}
    uint128 sum;
    uint128 x;
    word64 tt;
	for (int k = 0; k < l+1; ++k) {
		tt = tmp3[0];
        mult_64_64_128(tt, qHatModq[0][k], sum);
		for (int i = 1; i < K; ++i) {
			tt = tmp3[i];    
            x.lo=x.hi=0;
            mult_64_64_128(tt, pHatModq[i][k], x);
			sum += x;
		}
		ra[tid+(k<<log_degree)]+=k* degree * 23;
        barrett_reduction_128_64(sum, qVec[k], qrVec[k], qTwok[k]);
    }
    
	// for (int k = 0; k < K; ++k) {
	// 	tt = tmp3[0];
    //     mult_64_64_128(tt, pHatModq[0][k], sum);
	// 	for (int i = 0; i < K; ++i) {
	// 		tt = tmp3[i];
    //         x.lo=x.hi=0;
    //         mult_64_64_128(tt, pHatModq[i][k], x);
	// 		sum += x;
	// 	}
	// 	//ra[tid+((k+l+1)<<log_degree)]+=1234;
    //     barrett_reduction_128_64(sum, qVec[k], qrVec[k], qTwok[k]);
    // }
}
/*
//old version
__global__ void FastBasisConversionInModDown(   word64* a,      //input
                                                word64* ra,     //output
                                                int degree, int log_degree, int groupnums, 
                                                int alpha, int beta, int l, int K,
                                                word64* qVec, word64* qrVec, word64* qTwok,
                                                word64* pVec, word64* prVec, word64* pTwok,
                                                word64** qHatInvModq,
                                                word64** qHatModq,
                                                word64** pHatInvModp,
                                                word64** pHatModq
                                            ) 
{
    const int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int forn=degree/groupnums;
    word64* tmp3 = new word64[(l+1) << log_degree];
	for(int i = 0; i < l+1; ++i) {  //这部分是乘HatQ''^(-1) (mod q)(q就是这列对应的模数)
		word64* tmp3i = tmp3 + (i << log_degree);
		word64* ai = a + (i << log_degree);
		for(int n = tid*forn; n < (tid+1)*forn; ++n) {
            uint128 mult;
            mult_64_64_128(ai[n], qHatInvModq[l - 1][i], mult);
			tmp3i[n]=barrett_reduction_128_64(mult, qVec[i], qrVec[i], qTwok[i]);
		}
	}
	for (int k = 0; k < l+1; ++k) {
        //if(k/beta+1==index)continue;
		word64* rak = ra + ((k) << log_degree);
		for (int n = tid*forn; n < (tid+1)*forn; ++n) {//并行
			word64 tt = tmp3[n];
			uint128 sum;
            mult_64_64_128(tt, qHatModq[0][k], sum);
            uint128 x;
			for (int i = 1; i < l+1; ++i) {
				tt = tmp3[n + (i << log_degree)];    
                x.lo=x.hi=0;
                mult_64_64_128(tt, qHatModq[i][k], x);
				sum += x;
			}
			rak[n]=barrett_reduction_128_64(sum, qVec[k], qrVec[k], qTwok[k]);
		}
    }
    for(int i = 0; i < K; ++i) {  //这部分是乘HatQ'''^(-1) (mod q)(q就是这列对应的模数)
		word64* tmp3i = tmp3 + (i << log_degree);
		word64* ai = a + ((i+l+1) << log_degree);
		for(int n = tid*forn; n < (tid+1)*forn; ++n) {
            uint128 mult;
            mult_64_64_128(ai[n], pHatInvModp[K-1][i], mult);
			tmp3i[n]=barrett_reduction_128_64(mult, pVec[i], prVec[i], pTwok[i]);
		}
	}
	for (int k = 0; k < l+1; ++k) {
        //if(k/beta+1==index)continue;
		word64* rak = ra + ((k) << log_degree);
		for (int n = tid*forn; n < (tid+1)*forn; ++n) {//并行
			word64 tt = tmp3[n];
			uint128 sum;
            uint128 x;
            mult_64_64_128(tt, pHatModq[0][k], sum);
			for (int i = 0; i < K; ++i) {
				tt = tmp3[n + ((i+l+1) << log_degree)];
                x.lo=x.hi=0;
                mult_64_64_128(tt, pHatModq[i][k], x);
				sum += x;
			}
            sum+=uint128(rak[n]);
			rak[n]=barrett_reduction_128_64(sum, qVec[k], qrVec[k], qTwok[k]);
		}
    }
}
*/

__host__ void ModUp(    const int gridDim, const int blockDim,
                        word64* a,  //input
                        word64* ra, //output
                        int index,
                        int degree, int log_degree, int level, int alpha, int beta,
                        word64* qVec, word64* qInvVec,
                        word64* pVec, word64* pInvVec,
                        word64** qRootInvScalePows,
                        word64** qRootScalePows,
                        word64** pRootScalePows,
                        word64* qrVec, word64* qTwok,
                        word64* prVec, word64* pTwok,
                        word64** qHatInvModq,
                        word64** qHatModq,
                        word64** qHatModp
                    )
{
    //int groupnums=32*1024;
    int log_blockDim= 11;
    //for (int i=0;i<alpha;i++)
    //{
        
        qiNTTFront<<<256*alpha,128>>>(a, 
                                log_degree,
                                alpha,
                                index*alpha, degree,
                                qVec, qInvVec,
                                qRootInvScalePows
                                );
        /*
                        word64* a, 
                        int log_degree,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
        //printf("here!\n");
    //}    
        qiNTTBack<<<32*alpha,1024>>>(a, 
                                log_blockDim,
                                alpha,
                                index*alpha, degree,
                                qVec, qInvVec,
                                qRootInvScalePows
                                );
        /*
                        word64* a, 
                        int log_blockDim,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
    //}

/*
    word64* tmp3;
    cudaMalloc(&tmp3, sizeof(word64)*(alpha<<log_degree));
    //print("%d\n",alpha);
    //cudaMemset(tmp3, 0, sizeof(word64)*alpha *log_degree);
    cudaDeviceSynchronize();
*/

    FastBasisConversionInModUp<<<256, 256>>>( 
                                a,
                                ra,
                                index,
                                degree, log_degree,
                                alpha, beta, level, alpha,
                                //tmp3,
                                qVec, qrVec, qTwok,
                                pVec, prVec, pTwok,
                                qHatInvModq,
                                qHatModq,
                                qHatModp
                                );// alpha -> k+(l+1)
    
    // //cudaFree(tmp3);
    // /*
    //                             word64* a,      //input
    //                             word64* ra,     //output
    //                             int degree, int log_degree, int groupnums, 
    //                             int alpha, int beta, int l, int K,
    //                             word64* qVec, word64* qrVec, word64* qTwok,
    //                             word64* pVec, word64* prVec, word64* pTwok,
    //                             word64** qHatInvModq,
    //                             word64** qHatModq,
    //                             word64** qHatModp
    // */


    //for (int i=0;i<=level;i++)
    //{
    //    if(i>=index*alpha&&i<(index+1)*alpha)continue;
        qNTTFront<<<256*level,128>>>(ra, 
                                log_degree,
                                level,
                                0, degree,
                                qVec, qInvVec,
                                qRootScalePows
                                );
        /*
                        word64* a, 
                        int log_degree,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
        //printf("here!\n");
    //}
        qNTTBack<<<32*level,1024>>>(ra, 
                                log_blockDim,
                                level,
                                0, degree,
                                qVec, qInvVec,
                                qRootScalePows
                                );
        /*
                        word64* a, 
                        int log_blockDim,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
    
    //for (int i=0;i<alpha;i++)
    //{
        pNTTFront<<<256*alpha,128>>>(ra+((level+1)<<log_degree), 
                                log_degree,
                                alpha,
                                0, degree,
                                pVec, pInvVec,
                                pRootScalePows
                                );
        /*
                        word64* a, 
                        int log_degree,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
        //printf("here!\n");
    //}    
        pNTTBack<<<32*alpha,1024>>>(ra+((level+1)<<log_degree), 
                                log_blockDim,
                                alpha,
                                0, degree,
                                pVec, pInvVec,
                                pRootScalePows
                                );
        /*
                        word64* a, 
                        int log_blockDim,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */

}

__global__ void InnerProduct(   word64* a,          //input
                                uint128* res1, uint128* res2, //output
                                word64* evka, word64* evkb,
                                int index,
                                int degree, int log_degree, int l, int K, int alpha, int beta,
                                word64* qVec, word64* qrVec, word64* qTwok,
                                word64* pVec, word64* prVec, word64* pTwok
                            )
{
    const int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int nowpos;
    int delta=(l+1+K)*degree;
    uint128 x, y;
    uint128 sum1,sum2;
    /*
    word64 q=qVec[tid];
    word64 qr=qrVec[tid];
    word64 qT=qTwok[tid];
    word64 p=pVec[tid];
    word64 pr=prVec[tid];
    word64 pT=pTwok[tid];
    */
    //if(tid>=degree)
    //    printf("%d\n",tid);

    //word64 midres;
    /*
    for (int i=0; i<l+1; i++)//处理前面的
    {
        nowpos=i*degree+tid;
        //int topj=min(degree,(tid+1)*forn);
        {
            sum1.lo=sum1.hi=0;
            sum2.lo=sum2.hi=0;
            //mult_64_64_128(*(a+nowpos), *(evka+nowpos), sum1);
            //mult_64_64_128(*(a+nowpos), *(evkb+nowpos), sum2);
            for(int k=0; k<beta; k++)
            {
                x.lo=x.hi=0;
                mult_64_64_128(a[nowpos], evka[nowpos], x);
                //sum1+=x;
                res1[nowpos]+=barrett_reduction_128_64(x, qVec[i], qrVec[i], qTwok[i]);
                x.lo=x.hi=0;
                mult_64_64_128(a[nowpos], evkb[nowpos], x);
                //sum2+=x;
                res2[nowpos]=barrett_reduction_128_64(x, qVec[i], qrVec[i], qTwok[i]);
                nowpos+=delta;
            }
            
            //nowpos=i*degree+tid;
            //if(nowpos>(alpha+l+1)*degree)
            //    printf("@@@@%d\n",nowpos);
            //res1[nowpos]=
            //res1[nowpos]=barrett_reduction_128_64(sum1, q, qr, qT);//***
            //res1[nowpos]+=489124;
            //res2[nowpos]=
            //res2[nowpos]=barrett_reduction_128_64(sum2, q, qr, qT);//***
            //res2[nowpos]+=456; 
        }
    }
    for (int i=0; i<K; i++)//处理后面的
    {
        nowpos=((i+l+1)<<log_degree)+tid;
        {
            sum1.lo=sum1.hi=0;
            sum2.lo=sum2.hi=0;
            //mult_64_64_128(*(a+nowpos), *(evka+nowpos), sum1);
            //mult_64_64_128(*(a+nowpos), *(evkb+nowpos), sum2);
            for(int k=0; k<beta; k++)
            {
                //if(nowpos>(level+1+alpha)*beta*degree)
                //    printf("@@@@%d\n",nowpos);
                x.lo=x.hi=0;
                mult_64_64_128(a[nowpos], evka[nowpos], x);
                //sum1+=x;
                res1[nowpos]+=barrett_reduction_128_64(x, pVec[i], prVec[i], pTwok[i]);
                x.lo=x.hi=0;
                mult_64_64_128(a[nowpos], evkb[nowpos], x);
                //sum2+=x;
                res2[nowpos]+=barrett_reduction_128_64(x, pVec[i], prVec[i], pTwok[i]);
                nowpos+=delta;
            }
            //nowpos=((i+l+1)*degree)+tid;
            //res1[nowpos]=
            //res1[nowpos]=barrett_reduction_128_64(sum1, p, pr, pT);
            //res1[nowpos]+=123;
            //res2[nowpos]=
            //res2[nowpos]=barrett_reduction_128_64(sum2, p, pr, pT);
            //res2[nowpos]+=563;
        }
    }
    */
    //for(int k=0; k<beta; k++)
    //{
    int respos;
        for (int i=0; i<l+1; i++)//处理前面的
        {
            nowpos=i*degree+tid+index*delta;
            respos=i*degree+tid;
            sum1.lo=sum1.hi=0;
            sum2.lo=sum2.hi=0;
            x.lo=x.hi=0;
            mult_64_64_128(a[nowpos], evka[nowpos], x);
            if(index)
                res1[respos]+=x;
            else
                res1[respos]=x;
            y.lo=y.hi=0;
            mult_64_64_128(a[nowpos], evkb[nowpos], y);
            y.lo=evkb[nowpos];
            if(index)
                res2[respos]+=y;
            else
                res2[respos]=y;
        }

        for (int i=0; i<K; i++)//处理后面的
        {
            nowpos=((i+l+1)<<log_degree)+tid+index*delta;
            respos=((i+l+1)<<log_degree)+tid;
            {
                sum1.lo=sum1.hi=0;
                sum2.lo=sum2.hi=0;
                x.lo=x.hi=0;
                mult_64_64_128(a[nowpos], evka[nowpos], x);
                if(index)
                    res1[respos]+=x;
                else
                    res1[respos]=x;
                y.lo=y.hi=0;
                y=evkb[nowpos];
                if(index)
                    res2[respos]+=y;
                else
                    res2[respos]=y;
            }
        }
}

__global__ void InnerProductReduce(  
                                uint128* a1, uint128* a2, //output
                                word64* res1, word64* res2,
                                int degree, int log_degree, int l, int K,
                                word64* qVec, word64* qrVec, word64* qTwok,
                                word64* pVec, word64* prVec, word64* pTwok
                            )
{
    const int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int nowpos=tid;
    for(int i=0; i<=l ;i++)
    {  
        res1[nowpos]=barrett_reduction_128_64(a1[nowpos], qVec[i], qrVec[i], qTwok[i]);
        res2[nowpos]=barrett_reduction_128_64(a2[nowpos], qVec[i], qrVec[i], qTwok[i]);
        nowpos+=degree;
    }
    for(int i=0; i< K; i++)
    {
        res1[nowpos]=barrett_reduction_128_64(a1[nowpos], pVec[i], prVec[i], pTwok[i]);
        res2[nowpos]=barrett_reduction_128_64(a2[nowpos], pVec[i], prVec[i], pTwok[i]);
        nowpos+=degree;
    }
}

__host__ void ModDown(  const int gridDim, const int blockDim, 
                        word64* a,  //input
                        word64* ra, //output
                        int degree, int log_degree, int level, int alpha, int beta,
                        word64* qVec, word64* qInvVec,
                        word64* pVec, word64* pInvVec,
                        word64** qRootInvScalePows,
                        word64** pRootInvScalePows,
                        word64* qrVec, word64* qTwok,
                        word64* prVec, word64* pTwok,
                        word64** qHatInvModq,
                        word64** qHatModq,
                        word64** pHatInvModp,
                        word64** pHatModq,
                        word64** qRootScalePows
                        )
{
    //int groupnums=32*1024;
    int log_blockDim= 11;
    //for (int i=0; i<=level; i++)
    //{
    //     qiNTTFront<<<256*level,128>>>(a, 
    //                             log_degree,
    //                             level,
    //                             0, degree,
    //                             qVec, qInvVec,
    //                             qRootInvScalePows
    //                             );
    //     /*
    //                     word64* a, 
    //                     int log_degree,
    //                     int index, int degree,
    //                     word64* qVec, word64* qInvVec,
    //                     word64** qRootInvScalePows
    //     */
    // //}
    //     //printf("here!\n");
    //     qiNTTBack<<<32*level,1024>>>(a, 
    //                             log_blockDim,
    //                             level,
    //                             0, degree,
    //                             qVec, qInvVec,
    //                             qRootInvScalePows
    //                             );
        /*
                        word64* a, 
                        int log_blockDim,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
    //for (int i=0; i<alpha; i++)
    //{
        piNTTFront<<<256*alpha,128>>>(a+((level+1)<<log_degree), 
                                log_degree,
                                alpha,
                                0, degree,
                                pVec, pInvVec,
                                pRootInvScalePows
                                );
        /*
                        word64* a, 
                        int log_degree,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
    //}
        //printf("here!\n");
        piNTTBack<<<32*alpha,1024>>>(a+((level+1)<<log_degree), 
                                log_blockDim,
                                alpha,
                                0, degree,
                                pVec, pInvVec,
                                pRootInvScalePows
                                );
        /*
                        word64* a, 
                        int log_blockDim,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
    
    FastBasisConversionInModDown<<<256, 256>>>(   
                                    a,
                                    ra,
                                    degree, log_degree,
                                    alpha, beta, level, alpha,
                                    qVec, qrVec, qTwok,
                                    pVec, prVec, pTwok,
                                    qHatInvModq,
                                    qHatModq,
                                    pHatInvModp,
                                    pHatModq
                                );
    /*
        (   word64* a,      //input
            word64* ra,     //output
            int degree, int log_degree, int groupnums, 
            int alpha, int beta, int l, int K,
            word64* qVec, word64* qrVec, word64* qTwok,
            word64* pVec, word64* prVec, word64* pTwok,
            word64** qHatInvModq,
            word64** qHatModq,
            word64** pHatInvModp,
            word64** pHatModq
        ) 
    */
    
    //for (int i=0;i<=level;i++)
    //{
        qNTTFront<<<256*level,128>>>(a, 
                                log_degree,
                                level,
                                0, degree,
                                qVec, qInvVec,
                                qRootScalePows
                                );
        /*
                        word64* a, 
                        int log_degree,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */
    //}
        //printf("here!\n");
        qNTTBack<<<32*level,1024>>>(a, 
                                log_blockDim,
                                level,
                                0, degree,
                                qVec, qInvVec,
                                qRootScalePows
                                );
        /*
                        word64* a, 
                        int log_blockDim,
                        int index, int degree,
                        word64* qVec, word64* qInvVec,
                        word64** qRootInvScalePows
        */

    //求差
    // 不写了
}

__host__ void KeySwitch(word64* d2,  //d2 is in device 
                        int beta, int alpha, const int gridDim, const int blockDim,
                        word64* c0, word64* c1, //in device   output!
                        word64* qVec, word64* qrVec, word64* qTwok, //in device
                        word64* pVec, word64* prVec, word64* pTwok,
                        word64* evka, word64* evkb,
                        word64* qInvVec, word64* pInvVec,
                        word64**    qRootScalePows,     //(level+1)*degree
                        word64**    pRootScalePows,     //(alpha)*degree
                        word64**    qRootInvScalePows,  //(level+1)*degree
                        word64**    pRootInvScalePows,  //(alpha)*degree
                        word64**    qHatModq,           //(l+1)*(l+1)
                        word64**    qHatModp,           //(alpha__)*(l+1)
                        word64**    qHatInvModq,        //(l+1)*(l+1)
                        word64**    pHatInvModp,        //alpha_*alpha_
                        word64**    pHatModq        
                        )
{
    word64* ModUpResult;
    cudaMalloc(&ModUpResult, sizeof(word64) * ((level+1+alpha)*beta*degree));
    cudaDeviceSynchronize();
    word64* InnerProduct1;
    word64* InnerProduct2;
    cudaMalloc(&InnerProduct1, sizeof(word64) * ((level+1+alpha)*degree));
    cudaMalloc(&InnerProduct2, sizeof(word64) * ((level+1+alpha)*degree));

    uint128* InnerProductSum1;
    uint128* InnerProductSum2;
    //printf("size:%d\n",sizeof(uint128));
    cudaMalloc(&InnerProductSum1, sizeof(uint128) * ((level+1+alpha)*degree));
    cudaMalloc(&InnerProductSum2, sizeof(uint128) * ((level+1+alpha)*degree));


    for(int i=0; i<beta;i++)
    {
        ModUp(  gridDim, blockDim,
                d2+((i*alpha)<<log_degree),
                ModUpResult+((i*(alpha+level+1))<<log_degree) ,
                i,
                degree, log_degree, level, alpha, beta,
                qVec, qInvVec,
                pVec, pInvVec,
                qRootInvScalePows,
                qRootScalePows,
                pRootScalePows,
                qrVec, qTwok,
                prVec, pTwok,
                qHatInvModq,
                qHatModq,
                qHatModp
            );
        /*
        (   const int gridDim, const int blockDim 
            word64* a,  //input
            word64* ra, //output
            int degree, int log_degree, int level, int alpha, int beta,
            word64* qVec, word64* qInvVec,
            word64* pVec, word64* pInvVec,
            word64** qRootInvScalePows,
            word64** qRootScalePows,
            word64** pRootScalePows,
            word64* qrVec, word64* qTwok,
            word64* prVec, word64* pTwok,
            word64** qHatInvModq,
            word64** qHatModq,
            word64** qHatModp
        )
        */
    }


    cudaDeviceSynchronize();


    for(int i=0;i< beta;i++)
    {
        InnerProduct<<<256, 256>>>(   
                        ModUpResult,
                        InnerProductSum1, InnerProductSum2,
                        evka, evkb,
                        i,
                        degree, log_degree, level, alpha, alpha, beta,
                        qVec, qrVec, qTwok,
                        pVec, prVec, pTwok
                    );
    }

    InnerProductReduce<<<256, 256>>>(
                        InnerProductSum1, InnerProductSum2,
                        InnerProduct1, InnerProduct2,
                        degree, log_degree, level, alpha,
                        qVec, qrVec, qTwok,
                        pVec,  prVec, pTwok
    );

    cudaFree(InnerProductSum1);
    cudaFree(InnerProductSum2);
    //printf("InnerProduct finish!\n");
    /*
    (   word64* a,        //input
        word64* res1, word64* res2, //output
        word64* evka, word64* evkb,
        int groupnum,
        int degree, int log_degree, int l, int K, int alpha, int beta,
        word64* qVec, word64* qrVec, word64* qTwok,
        word64* pVec, word64* prVec, word64* pTwok
    )
    */

    ModDown(gridDim, blockDim,
            InnerProduct1,
            c0,
            degree, log_degree, level, alpha, beta,
            qVec, qInvVec,
            pVec, pInvVec,
            qRootInvScalePows,
            pRootInvScalePows,
            qrVec, qTwok,
            prVec, pTwok,
            qHatInvModq,
            qHatModq,
            pHatInvModp,
            pHatModq,
            qRootScalePows
            );
    ModDown(gridDim, blockDim,
            InnerProduct2,
            c1,
            degree, log_degree, level, alpha, beta,
            qVec, qInvVec,
            pVec, pInvVec,
            qRootInvScalePows,
            pRootInvScalePows,
            qrVec, qTwok,
            prVec, pTwok,
            qHatInvModq,
            qHatModq,
            pHatInvModp,
            pHatModq,
            qRootScalePows
            );
//     /*
//     (   const int gridDim, const int blockDim 
//         word64* a,  //input
//         word64* ra, //output
//         int degree, int log_degree, int level, int alpha, int beta,
//         word64* qVec, word64* qInvVec,
//         word64* pVec, word64* pInvVec,
//         word64** qRootInvScalePows,
//         word64** pRootInvScalePows,
//         word64* qrVec, word64* qTwok,
//         word64* prVec, word64* pTwok,
//         word64** qHatInvModq,
//         word64** qHatModq,
//         word64** pHatInvModp,
//         word64** pHatModq,
//         word64** qRootScalePows
//     )
//     */
//    /*
//     cudaEventRecord(stop,0);
//     cudaEventSynchronize(start);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&time_elapsed,start,stop);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     printf("time:%fms\n",time_elapsed);
//     */
}

__global__ void gpu_tensor_product( word64* c1_a, word64* c1_b, 
                                    word64* c2_a, word64* c2_b, 
                                    int* glog_degree,
                                    int* gdegree, int* glength,
                                    word64* gprimes,
                                    word64* gbarret_ratio, uint32_t* gbarret_k,
                                    word64* d0,
                                    word64* d1,
                                    word64* d2)
{
    uint128 out1, out2, out3, out4;
    uint128 temp1, temp2;
    int prime_idx;
    word64 prime;
    //if(threadIdx.x==0) 
    //    printf("%d %d %d %d\n",gdegree[0], glength[0], blockDim.x, gridDim.x);
    for(int tid=threadIdx.x+blockIdx.x*blockDim.x; tid< gdegree[0]*glength[0] ; tid+=blockDim.x*gridDim.x)
    {
        prime_idx = tid >> glog_degree[0];
        prime=gprimes[prime_idx];
        out1.lo=out1.hi=0;
        out2.lo=out2.hi=0;
        out3.lo=out3.hi=0;
        out4.lo=out4.hi=0;
        mult_64_64_128(c1_a[tid], c2_a[tid], out1);
        mult_64_64_128(c1_b[tid], c2_b[tid], out2);
		//printf("%lld %lld\n",c1_b[tid], c2_b[tid]);
        mult_64_64_128(c1_a[tid], c2_b[tid], out3);
        mult_64_64_128(c1_b[tid], c2_a[tid], out4);

        auto gbarret_ratiox=gbarret_ratio[prime_idx];
        auto gbarret_kx=gbarret_k[prime_idx];
        word64 op_out1 = barrett_reduction_128_64(out1, prime , gbarret_ratiox
                                    ,gbarret_kx, temp1, temp2);
        word64 op_out2 = barrett_reduction_128_64(out2, prime , gbarret_ratiox
                                    ,gbarret_kx, temp1, temp2);
        word64 op_out3 = barrett_reduction_128_64(out3, prime , gbarret_ratiox
                                    ,gbarret_kx, temp1, temp2);
        word64 op_out4 = barrett_reduction_128_64(out4, prime , gbarret_ratiox
                                    ,gbarret_kx, temp1, temp2);

        //printf("#### %lld %lld %lld %lld \n",op_out1,op_out2,op_out3,op_out4);
        d0[tid]=op_out2;
        d1[tid]=op_out3+op_out4;
        if(prime-d1[tid]>>63)d1[tid]-=prime;
        d2[tid]=op_out1;
	//d0[tid]=c1_a[tid];
    }                     
}

//simulate computer rand process
static word64 seed=19260817;
word64 randword64()
{
    return seed=seed*seed*seed*114514+seed*seed*20230221-seed*20221115+19;
}

using namespace std; 

void init()
{
    degree=1<<log_degree;
    chain_length=level+1;
    alpha_=(level+1)/dnum;
    l_=level;
    beta_=(l_+1)/alpha_;
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
    word64 *d2;
    auto sumlength=degree * chain_length;
    c1_a = (word64*)malloc(sizeof(word64) * sumlength);
    c1_b = (word64*)malloc(sizeof(word64) * sumlength);
    c2_a = (word64*)malloc(sizeof(word64) * sumlength);
    c2_b = (word64*)malloc(sizeof(word64) * sumlength);
    d0 = (word64*)malloc(sizeof(word64) * sumlength);
    d1 = (word64*)malloc(sizeof(word64) * sumlength);
    d2 = (word64*)malloc(sizeof(word64) * sumlength);

    //*******
    //生成模数
    //*******

    //1 CPU生成
    word64 *barret_ratio;
    uint32_t *barret_k;
    word64 *barret_k_64;
    barret_ratio= (word64*)malloc(sizeof(word64*)*max_num_moduli);
    barret_k    = (uint32_t*)malloc(sizeof(uint32_t)*max_num_moduli);
    barret_k_64 = (word64*)malloc(sizeof(word64*)*max_num_moduli);
    for (auto i=0; i<max_num_moduli; i++){
        auto p=primes[i];
        uint32_t barret = floor(log2(p)) + 63;
        barret_k[i]=barret;
        barret_k_64[i]=barret;
        __int128 temp = ((uint64_t)1 << (barret - 64));
        temp <<= 64;
        temp = temp/p;
        barret_ratio[i]=(uint64_t(temp));
        //printf("***");
        //print(temp);
        //printf("\n");
    } 

    word64*     qInvVec;                                //这不会算，全rand了
    word64*     pInvVec;
    word64**    qRootScalePows;     //(level+1)*degree
    word64**    pRootScalePows;     //(alpha)*degree
    word64**    qRootInvScalePows;  //(level+1)*degree
    word64**    pRootInvScalePows;  //(alpha)*degree
    word64**    qHatModq;           //(l+1)*(l+1)
    word64**    qHatModp;           //(alpha__)*(l+1)
    word64**    qHatInvModq;        //(l+1)*(l+1)
    word64**    pHatInvModp;        //alpha_*alpha_
    word64**    pHatModq;           //alpha * (level+1)
    qInvVec             =(word64*)  malloc(sizeof(word64) * (level+1));
    pInvVec             =(word64*)  malloc(sizeof(word64) * (alpha_));
    qRootScalePows      =(word64**) malloc(sizeof(word64*) * (level+1));
    pRootScalePows      =(word64**) malloc(sizeof(word64*) * (alpha_));
    qRootInvScalePows   =(word64**) malloc(sizeof(word64*) * (level+1));
    pRootInvScalePows   =(word64**) malloc(sizeof(word64*) * (alpha_));
    qHatModq            =(word64**) malloc(sizeof(word64*) * (level+1));
    qHatModp            =(word64**) malloc(sizeof(word64**) * (alpha_));
    qHatInvModq         =(word64**) malloc(sizeof(word64*) * (level+1));
    pHatInvModp         =(word64**) malloc(sizeof(word64*) * (alpha_));
    pHatModq            =(word64**) malloc(sizeof(word64**) * (alpha_));
    for (int i=0; i<(level+1);i++)
    {
        word64 p=primes[i];
        qInvVec[i]=randword64()%p;
    }
    for (int i=0; i<(alpha_);i++)
    {
        word64 p=primes[level+1+i];
        pInvVec[i]=randword64()%p;
    }

    for (int i=0; i<(level+1); i++)
    {
        qRootScalePows[i]=(word64*)malloc(sizeof(word64) * (degree));
        qRootInvScalePows[i]=(word64*)malloc(sizeof(word64) * (degree));
        word64 p=primes[i];
        for (int j=0; j<degree; j++)
        {
            qRootScalePows[i][j]=randword64()%p;
            qRootInvScalePows[i][j]=randword64()%p;
        }
    }
    for (int i=0; i<(alpha_); i++)
    {
        pRootScalePows[i]=(word64*)malloc(sizeof(word64) * (degree));
        pRootInvScalePows[i]=(word64*)malloc(sizeof(word64) * (degree));
        word64 p=primes[i+level+1];
        for (int j=0; j<degree; j++)
        {
            pRootScalePows[i][j]=randword64()%p;
            pRootInvScalePows[i][j]=randword64()%p;
        }
    }

//    
//        qHatModq[i]=(word64**)malloc(sizeof(word64*) * (level+1));
        for(int j=0; j<(level+1);j++)
        {
            qHatModq[j]=(word64*)malloc(sizeof(word64) * (level+1));
            word64 p=primes[j];  
            for (int k=0; k<(level+1);k++)
            {
                qHatModq[j][k]=randword64()%p;
            }    
        }
        
//        qHatModp[i]=(word64**)malloc(sizeof(word64*) * (alpha_));
        for(int j=0; j<(alpha_);j++)
        {
            qHatModp[j]=(word64*)malloc(sizeof(word64) * (level+1));
            word64 p=primes[j+level+1];
            for (int k=0; k<(level+1);k++)
            {
                qHatModp[j][k]=randword64()%p;
            }
        }
    for (int i=0; i<(level+1); i++)
    {
        qHatInvModq[i]=(word64*)malloc(sizeof(word64) * (level+1));
        for(int j=0; j<(level+1);j++)
        {
            word64 p=primes[i];
            qHatInvModq[i][j]=randword64()%p;
        }
    }
    
    for(int i=0; i<alpha_;i++)
    {
        pHatInvModp[i]=(word64*)malloc(sizeof(word64) * (alpha_));
        for(int j=0; j<alpha_; j++)
        {
            word64 p=primes[level+1+j];
            pHatInvModp[i][j]=randword64()%p;
        }
    }
    for(int j=0; j<alpha_; j++)
        {
            pHatModq[j]=(word64*)malloc(sizeof(word64) * (level+1));
            for(int k=0; k<(level+1);k++)
            {
                word64 p=primes[k];
                pHatModq[j][k]=randword64()%p;
            }
        }

    //2 GPU内存分配
    word64* qVec;                                   //模数q
    word64* pVec;                                   //模数p
    word64 *qrVec, *qTwok;                          //q barret mod用参数
    word64 *prVec, *pTwok;                          //p barret mod用参数
    cudaMalloc(&qVec, sizeof(word64) * (level+1));  //q长度l+1
    cudaMalloc(&pVec, sizeof(word64) * (alpha_));   //长度(L+1)/dnum=alpha
    cudaMalloc(&qrVec, sizeof(word64) * (level+1)); //同q
    cudaMalloc(&qTwok, sizeof(word64) * (level+1));
    cudaMalloc(&prVec, sizeof(word64) * (alpha_));  //同p
    cudaMalloc(&pTwok, sizeof(word64) * (alpha_));
    word64*     gqInvVec;                                
    word64*     gpInvVec;
    word64**    gqRootScalePows;     //(level+1)*degree
    word64**    gpRootScalePows;     //(alpha)*degree
    word64**    gqRootInvScalePows;  //(level+1)*degree
    word64**    gpRootInvScalePows;  //(alpha)*degree
    word64**    gqHatModq;           //(l+1)*(l+1)
    word64**    gqHatModp;           //(alpha__)*(l+1)
    word64**    gqHatInvModq;        //(l+1)*(l+1)
    word64**    gpHatInvModp;        //alpha_*alpha_
    word64**    gpHatModq;           //alpha * (level+1)
    cudaMalloc(&gqInvVec, sizeof(word64) * (level+1));
    cudaMalloc(&gpInvVec, sizeof(word64) * (alpha_));
    cudaMalloc(&gqRootScalePows, sizeof(word64) * (level+1) * degree);
    cudaMalloc(&gpRootScalePows, sizeof(word64) * (alpha_) * degree);
    cudaMalloc(&gqRootInvScalePows, sizeof(word64) * (level+1) * degree);
    cudaMalloc(&gpRootInvScalePows, sizeof(word64) * (alpha_) * degree);
    cudaMalloc(&gqHatModq, sizeof(word64) * (level+1) * (level+1));
    cudaMalloc(&gqHatModp, sizeof(word64) * (alpha_) * (level+1));
    cudaMalloc(&gqHatInvModq, sizeof(word64) * (level+1) * (level+1));
    cudaMalloc(&gpHatInvModp, sizeof(word64) * (alpha_) * (alpha_));
    cudaMalloc(&gpHatModq, sizeof(word64) * (alpha_) * (level+1));
    //3 拷贝至GPU
    cudaMemcpy(qVec, primes, sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
    cudaMemcpy(qrVec, barret_ratio, sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
    cudaMemcpy(qTwok, barret_k_64, sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
    cudaMemcpy(pVec, primes+(level+1), sizeof(word64) * (alpha_), cudaMemcpyHostToDevice);
    cudaMemcpy(prVec, barret_ratio+(level+1), sizeof(word64) * (alpha_), cudaMemcpyHostToDevice);
    cudaMemcpy(pTwok, barret_k_64+(level+1), sizeof(word64) * (alpha_), cudaMemcpyHostToDevice);
    cudaMemcpy(gqInvVec, qInvVec, sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpInvVec, pInvVec, sizeof(word64) * (alpha_), cudaMemcpyHostToDevice);


    for (int i=0; i<(level+1); i++)
    {
        //cudaMalloc(&gqRootScalePows[i], sizeof(word64) * (degree));
        //cudaMalloc(&gqRootInvScalePows[i], sizeof(word64) * (degree));
        cudaMemcpy(gqRootScalePows+i*degree, qRootScalePows[i], sizeof(word64) * (degree), cudaMemcpyHostToDevice);
        cudaMemcpy(gqRootInvScalePows+i*degree, qRootInvScalePows[i], sizeof(word64) * (degree), cudaMemcpyHostToDevice);
    }
    
    for (int i=0; i<(alpha_); i++)
    {
        //cudaMalloc(&gpRootScalePows[i], sizeof(word64) * (degree));
        //cudaMalloc(&gpRootInvScalePows[i], sizeof(word64) * (degree));
        cudaMemcpy(gpRootScalePows+i*degree, pRootScalePows[i], sizeof(word64) * (degree), cudaMemcpyHostToDevice);
        cudaMemcpy(gpRootInvScalePows+i*degree, pRootInvScalePows[i], sizeof(word64) * (degree), cudaMemcpyHostToDevice);
    }
    for(int j=0; j<(level+1);j++)
        {
            //cudaMalloc(&gqHatModq[j], sizeof(word64) * (level+1));
            cudaMemcpy(gqHatModq+j*(level+1), qHatModq[j], sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
        }
    
    for(int j=0; j<(alpha_);j++)
        {
            //cudaMalloc(&gqHatModp[j], sizeof(word64) * (level+1));
            cudaMemcpy(gqHatModp+j*(level+1), qHatModp[j], sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
        }

    for (int i=0; i<(level+1); i++)
    {
        //cudaMalloc(&gqHatInvModq[i], sizeof(word64) * (level+1));
        cudaMemcpy(gqHatInvModq+i*(level+1), qHatInvModq[i], sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<alpha_;i++)
    {
        //cudaMalloc(&gpHatInvModp[i], sizeof(word64) * (alpha_));
        cudaMemcpy(gpHatInvModp+i*alpha_, pHatInvModp[i], sizeof(word64) * (alpha_), cudaMemcpyHostToDevice);  
    }
    for(int j=0; j<alpha_; j++)
        {
            //cudaMalloc(&gpHatModq[j], sizeof(word64) * (level+1));
            cudaMemcpy(gpHatModq+j*(level+1), pHatModq[j], sizeof(word64) * (level+1), cudaMemcpyHostToDevice);
        }
    //*********
    //生成辅助密钥evk
    //*********
    
    //1 CPU生成
    word64* evka;
    word64* evkb;
    evka=(word64*)malloc(sizeof(word64*)*(beta_*max_num_moduli*degree));
    evkb=(word64*)malloc(sizeof(word64*)*(beta_*max_num_moduli*degree));
    int now=0;
    for (int i=0; i<beta_; i++)
    {
        for (int j=0; j<max_num_moduli;j++)
        {
            auto prime=primes[j];
            for (int k=0; k<degree; k++)
            {
                evka[now]=randword64()%prime;
                evkb[now]=randword64()%prime;
            }
        }
    }
    //2 GPU内存分配
    word64* gevka;
    word64* gevkb;
    cudaMalloc(&gevka, sizeof(word64) * (beta_*max_num_moduli*degree));
    cudaMalloc(&gevkb, sizeof(word64) * (beta_*max_num_moduli*degree));
    //3 拷贝至GPU
    cudaMemcpy(gevka, evka, sizeof(word64) * (beta_*max_num_moduli*degree), cudaMemcpyHostToDevice);
    cudaMemcpy(gevkb, evkb, sizeof(word64) * (beta_*max_num_moduli*degree), cudaMemcpyHostToDevice);
    
    //*********
    //生成随机多项式
    //*********
    now=0;
    for (auto i = 0; i < level+1; i++) {
        auto prime=primes[i];
        for(auto j = 0; j< degree; j++){
            c1_a[now] = randword64()%prime;
            c1_b[now] = randword64()%prime;
            c2_a[now] = randword64()%prime;
            c2_b[now] = randword64()%prime;
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
    word64 *gd2;
    cudaMalloc(&gd0, sizeof(word64) * sumlength);
    cudaMalloc(&gd1, sizeof(word64) * sumlength);
    cudaMalloc(&gd2, sizeof(word64) * sumlength);

    word64 *gprimes;
    cudaMalloc(&gprimes, sizeof(word64) * max_num_moduli);

    word64 *gbarret_ratio;
    uint32_t *gbarret_k;
    cudaMalloc(&gbarret_ratio, sizeof(unsigned long long) * max_num_moduli);
    cudaMalloc(&gbarret_k, sizeof(uint32_t) * max_num_moduli);

    int* gdegree, *glength, *glog_degree;
    cudaMalloc(&gdegree, sizeof(int));
    cudaMalloc(&glength, sizeof(int) );
    cudaMalloc(&glog_degree, sizeof(int));

    int *degree_1,*chain_length_1,*log_degree_1; 
    degree_1=(int*)malloc(sizeof(int)*1);
    chain_length_1=(int*)malloc(sizeof(int));
    log_degree_1=(int*)malloc(sizeof(int));
	
	//printf("%ld %ld %d\n",sizeof(*degree_12),sizeof((int)degree),degree_12[0]);
    degree_1[0]=(int)degree;
	chain_length_1[0]=(int)chain_length;
	log_degree_1[0]=log_degree;
    //printf("%d %d %d\n",degree, chain_length, log_degree);

    // copy polynomial from host to device memory
    cudaMemcpy(gc1_a, c1_a, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gc1_b, c1_b, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gc2_a, c2_a, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gc2_b, c2_b, sizeof(word64) * sumlength, cudaMemcpyHostToDevice);
    cudaMemcpy(gprimes, primes, sizeof(word64) * max_num_moduli, cudaMemcpyHostToDevice);
    cudaMemcpy(gdegree, degree_1, sizeof(int) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(glength, chain_length_1, sizeof(int) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(glog_degree, log_degree_1, sizeof(int) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(gbarret_ratio, barret_ratio, sizeof(word64) * max_num_moduli, cudaMemcpyHostToDevice);
    //cudaMemcpy(gbarret_k, barret_k, sizeof(uint32_t) * max_num_moduli, cudaMemcpyHostToDevice);
	//printf("12312312312312312");

    
    float time_elapsed;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


    //tensor-product
    int gridDim = 32;
    int blockDim = 1024;
    printf("%d %d\n", gridDim, degree);
    
    gpu_tensor_product<<<512, 128>>> ( gc1_a, gc1_b, gc2_a, gc2_b, 
                                                glog_degree, gdegree, glength, gprimes, 
                                                gbarret_ratio, gbarret_k,
                                                gd0, gd1, gd2);
    
    //分配c0,c1空间
    word64* gc0;
    word64* gc1;
    word64* c0;
    word64* c1;
    c0=(word64*)malloc(sizeof(word64*)*((level+1)*degree));
    c1=(word64*)malloc(sizeof(word64*)*((level+1)*degree));
    cudaMalloc(&gc0, sizeof(word64) * (level+1)*degree);
    cudaMalloc(&gc1, sizeof(word64) * (level+1)*degree);

    //printf("beta:%d alpha:%d\n", beta_, alpha_);

    //key-switch
    KeySwitch(  gd2, 
                beta_, alpha_, gridDim, blockDim, 
                gc0, gc1,
                qVec, qrVec, qTwok,
                pVec, prVec, pTwok,
                gevka, gevkb,
                gqInvVec, gpInvVec,
                gqRootScalePows, gpRootScalePows, gqRootInvScalePows, gpRootInvScalePows,
                gqHatModq, gqHatModp,
                gqHatInvModq, gpHatInvModp,
                gpHatModq
                );


    cudaEventRecord(stop,0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("time:%fms\n",time_elapsed);



    cudaMemcpy(d0, gd0, sizeof(word64) * sumlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(d1, gd1, sizeof(word64) * sumlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(d2, gd2, sizeof(word64) * sumlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(c0, gc0, sizeof(word64) * sumlength, cudaMemcpyDeviceToHost);
    cudaMemcpy(c1, gc1, sizeof(word64) * sumlength, cudaMemcpyDeviceToHost);
//sleep(5);
	
    //cout<<"?????"<<endl;
    //printf("%ld\n",sizeof(ans_a));
    /*=
    for (int i = 0; i < sumlength; ++i) {
        printf("%llu %llu %llu\n",d0[i],d1[i],d2[i]);
    }
    printf("\n");
    */
/*
    for (int i = 0; i < sumlength; ++i) {
        printf("%lld\t",d1[i]);
    }
    printf("\n");
    for (int i = 0; i < sumlength; ++i) {
        printf("%lld\t",d2[i]);
    }
    printf("\n");
*/
    cudaFree(gc1_a);
    cudaFree(gc1_b);
    cudaFree(gc2_a);
    cudaFree(gc2_b);
    cudaFree(&gprimes);
    cudaFree(&gdegree);
    cudaFree(&glength);
    cudaFree(&glog_degree);

    cudaFreeHost(c1_a);
    cudaFreeHost(c1_b);
    cudaFreeHost(c2_a);
    cudaFreeHost(c2_b);
    cudaFreeHost(d0);
    cudaFreeHost(d1);
    cudaFreeHost(d2);
    return 0;
}
