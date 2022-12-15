//dynamical matrix A

#include <math.h>

#include <stdint.h>

#include <stdio.h>



//parameters are: (ns), (Phi), Q, R ,  >>>lambda <<<, scales and coefs


int16_t ns = 17; //nstates
int16_t Phi[19] = {1962, -503,  503, 1962, 1023,  255,  127,   79,   55,   41,   32,
         26,   22,   18,   16,   14,   12,   11,   10};
          
   
       
int16_t Q[3] = {32752, 32752, 2047};

int16_t R = 0;
int32_t x[17] = {0};//posterior state
int32_t x_[17] = {0};//posterior state
int32_t P[289] = {0};
int32_t P_[289] = {0};
//int32_t P_buf[ns*ns] = {0};

int32_t sigma = 0;

int32_t K[17]  = {0};
int32_t n[17] = {0};

int cc = 0;
int cc2 = 0;

int32_t no = 0;

int16_t first_scale = 0x07FF;
int16_t second_scale = 0x0004;
int16_t coef = 0x07FF/0x0004;



//instead of it load all the coeficients


// end of the init (it is //)
int16_t y[20] = {1384,  1035,  -128,   677,  -746,   880,   717,   319,  3250, 4825,  4338,  5820,  5286,  3907,  5387,  2406,  1726,   237,  -1557, -1910};
//{538,  620, -784 , 208 ,-899,   91  ,775 , 147 ,2906 ,4147, 3791 ,5788, 5226 ,4057, 6316, 4427, 4316 ,4046, 2985, 3574};
//{347,  1375 ,  541, -1113 , 72,  2804,2883, 1215, 2609 , 5378 , 4294,  7296,6644,  6246,  8473 , 6840, 6224,  7532,  6071,  6636};




int main(void)
{

for (int ext; ext < 20; ext ++)
{
	//prediction x
	x_[0] = (Phi[0]*x[0]+Phi[1]*x[1])/first_scale;
	x_[1] = (Phi[2]*x[0]+Phi[3]*x[1])/first_scale;
	
	
	x_[2] = 0;	
	for (cc = 2; cc < ns; cc++)
	{
		x_[2] += Phi[2+cc]*x[cc];
		
	}
	x_[2] /= first_scale;
	for(cc = 3; cc < ns; cc++)
	{
		x_[cc] = x[cc-1];				
	}
	
	//printf("%d,%d,%d,%d,%d,%d,%d,",x_[0],x_[1],x_[2],x_[3],x_[4],x_[5],x_[6]);
	
	//for (int j = 0; j < 10; j++)
	//{
	//	for (int i =0; i<10; i++)
	//	{
	//		printf("%d,",P[j*ns+i]);
	//	}
	//	printf("\n");
	//}
	//prediction P
	// may be to do symmetricity check
	for(cc2 = 0; cc2 < ns; cc2++)
	{
		P_[0*ns+cc2] = (Phi[0]*P[0*ns+cc2]+Phi[1]*P[1*ns+cc2])/first_scale;
		P_[1*ns+cc2] = (Phi[2]*P[0*ns+cc2]+Phi[3]*P[1*ns+cc2])/first_scale;
		
		P_[2*ns+cc2] = 0;	
		for (cc = 2; cc < ns; cc++)
		{
			P_[2*ns+cc2] += Phi[2+cc]*P[cc*ns+cc2];
			
		}
		P_[2*ns+cc2] /= first_scale;
		for(cc = 3; cc < ns; cc++)
		{
			P_[cc*ns+cc2] = P[(cc-1)*ns+cc2];
					
		}
	}
	
	//for (int j = 0; j < 10; j++)
	//{
	//	for (int i =0; i<10; i++)
	//	{
	//		printf("%d,",P_[j*ns+i]);
	//	}
	//	printf("\n");
	//}
	
	//printf("%d,%d,%d,\n%d,%d,%d,\n%d,%d,%d,",P_[0],P_[1],P_[2],P_[102+0],P_[102+1],P_[102+2],P_[102*2+0],P_[102*2+1],P_[102*2+2]);
	

	for(cc2 = 0; cc2 < ns; cc2++)
	{

		
		P[0+cc2*ns] = (Phi[0]*P_[0+cc2*ns]+Phi[1]*P_[1+cc2*ns])/second_scale;
		P[1+cc2*ns] = (Phi[2]*P_[0+cc2*ns]+Phi[3]*P_[1+cc2*ns])/second_scale;
		
		//  STOPPED	
		
		P[2+cc2*ns] = 0;
		
		for (cc = 2; cc < ns; cc++)
		{
			P[2+cc2*ns] += Phi[2+cc]*P_[cc2*ns+cc];
			
		}
		P[2+cc2*ns] /= second_scale;
		for(cc = 3; cc < ns; cc++)
		{
			P[cc+cc2*ns] = P_[cc2*ns+cc+1]*first_scale/second_scale;				
		}
	}
	
	
	
	P[0] += Q[0]/second_scale;
	P[1+ns] += Q[1]/second_scale;
	P[2+2*ns] += Q[2]/second_scale;
	
	//printf("%d,%d,%d,\n%d,%d,%d,\n%d,%d,%d,",P[0],P[1],P[2],P[102+0],P[102+1],P[102+2],P[102*2+0],P[102*2+1],P[102*2+2]);
	
	
	
	sigma = (P[0]+P[2*ns]+P[2]+P[2*ns+2]+R)/coef;
	
	if (sigma == 0)
	{
		sigma = 1;
	}
	
	//printf("%d\n",sigma);
	
	no = y[ext]-x_[0]-x_[2]; // DELETE IDX	
	for(cc = 0;cc<ns;cc++)
	{
		K[cc] = (P[cc*ns]+P[cc*ns+2])/sigma;
		x[cc] = x_[cc]+(no*K[cc])/coef;
		
	}
	
	//printf("%d, %d,%d, %d,%d, %d,%d, %d\n",K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7]);
	//printf("%d, %d,%d, %d,%d, %d,%d, %d\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
	
	
	//printf("%d, %d,%d, %d,%d, %d,%d, %d\n",P[0],P[1],P[2],P[0+ns],P[1+ns],P[2+ns],P[0+ns*2],P[1+ns*2]);
	

	
	
	for(cc=  0; cc<ns;cc++)
	{
	
		for (cc2= 0; cc2<ns; cc2++)
		{
			P_[cc2+cc*ns] = (coef*P[cc2+cc*ns]-K[cc]*P[cc2]-K[cc]*P[cc2+2*ns])/(coef*coef);
		}
	}
	
	
	for(cc=  0; cc<ns;cc++)
	{
		for (cc2= 0; cc2<ns; cc2++)
		{
			P[cc2+cc*ns] = P_[cc2+cc*ns];
		}
	}  /// TO OPTIMIZE IT 
	
	//printf("%d, %d,%d, %d,\n%d, %d,%d, %d,\n%d, %d,%d, %d,\n%d, %d,%d, %d\n",P[0],P[1],P[2],P[3],P[0+ns],P[1+ns],P[2+ns],P[3+ns],P[0+2*ns],P[1+2*ns],P[2+2*ns],P[3+2*ns],P[0+3*ns],P[1+3*ns],P[2+3*ns],P[3+3*ns]);
	
	
	printf("%d\n",x[0]);
}
}