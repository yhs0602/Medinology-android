/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <string.h>
#include <jni.h>
#include "/storage/emulated/0/AppProjects/Medinology3/jni/eigen/Eigen/Dense"
#include <iostream>
#include <fstream>
#include<stdio.h>
using namespace Eigen;
using namespace std;
ofstream logger("/sdcard/predictlog.txt");

static void Do();
double identity_function(double x);
double step_function(double x);
double sigmoid(double x);
double sigmoid_grad(double x);
double relu(double x);
double expfunc(double x);
//FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!
double relu_grad(double x);
MatrixXd identityFunction(MatrixXd x);
MatrixXd stepFunction(MatrixXd x);
MatrixXd Sigmoid(MatrixXd x);
MatrixXd Sigmoid_Grad(MatrixXd x);
MatrixXd Softmax(MatrixXd x);
void buildNormalDist(MatrixXd dest,int sx,int sy);

void LoadWeights(const char *filename);



double identity_function(double x)
{
	return x;
}

double step_function(double x)
{
	if(x>0)
		return 1;
	return 0;
}

double sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

double sigmoid_grad(double x)
{
	double s=sigmoid(x);
	return (1.0-s)*s;
}

double relu(double x)
{
	if(x>0)return x;
	return 0.0;
}

//FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!
double relu_grad(double x)
{
	return relu(x);
}

double expfunc(double x)
{
	return exp(x);
}

MatrixXd IdentityFunction(MatrixXd x)
{
	return x;
}

MatrixXd StepFunction(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&step_function);
	return m;
}

MatrixXd Sigmoid(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&sigmoid);
	return m;
}

MatrixXd Sigmoid_Grad(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&sigmoid_grad);
	return m;
}

MatrixXd Relu(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&relu);
	return m;
}
/*
MatrixXd Relu_grad()
{
	MatrixXd m=MatrixXf::Zero();
	m.unaryExpr(&relu);
	return m;	
}
*/
MatrixXd Softmax(MatrixXd x)
{
	MatrixXd y=x.array()-x.maxCoeff();
	y.unaryExpr(&expfunc);
	double s=y.sum();
	return y/s;
}

float cross_entropy_error(MatrixXd y,MatrixXd t)
{
	int i,j;
	t.maxCoeff(&i,&j);
	return -log(y(0,j));
}

class Layer
{
	public:
	virtual MatrixXd forward(MatrixXd x)=0;
	virtual MatrixXd backward(MatrixXd dout)=0;
};
class SigmoidLayer:public Layer
{
	public:
	MatrixXd forward(MatrixXd x)
	{
		out=Sigmoid(x);
		return out;
	}
	MatrixXd backward(MatrixXd dout)
	{
		MatrixXd dx=Sigmoid_Grad(out);
	    dx*=dout(0,0);
		return dx;
	}
	MatrixXd out;
};

/*
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
*/
class AffineLayer:public Layer
{
	public:
	MatrixXd forward(MatrixXd px)
	{
		x=px;
		MatrixXd out=x*W+b;
		return out;
	}
	MatrixXd backward(MatrixXd dout)
	{
		MatrixXd dx=dout*(W.transpose());
		dW=(x.transpose())*dout;
	//	db=dout.sum();
		return dx;
	}
	AffineLayer(MatrixXd pW,MatrixXd pb)
	{
		W=pW;
		b=pb;
	}
	MatrixXd x,W,b,dW,db;
};
/*

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
*/
class SoftmaxWithLossLayer:public Layer
{
	public:
	MatrixXd forward(MatrixXd x,MatrixXd pt)
	{
		t=pt;
		y=Softmax(x);
		loss=MatrixXd::Constant(1,1,cross_entropy_error(y,t));
		return loss;
	}
	MatrixXd backward(MatrixXd dout)
	{
		MatrixXd dx=y.array()-t.array();
		return dx;
	}
	SoftmaxWithLossLayer()
	{
		
	}
	MatrixXd forward(MatrixXd x){
		
	}
	MatrixXd y,t,loss;
};
/*

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

		*/
		
	class TwoLayerNet
	{
		public:
		MatrixXd W1,W2,b1,b2;
		MatrixXd gradW1,gradW2,gradb1,gradb2;
		Layer *layers[3];
		Layer *lastLayer;
		TwoLayerNet(int inputsize,int hiddensize,int outputsize)
		{
			W1=MatrixXd(inputsize,hiddensize);
			W2=MatrixXd(hiddensize,outputsize);
			b1=MatrixXd::Zero(1,hiddensize);
			b2=MatrixXd::Zero(1,outputsize);
			//buildNormalDist(W1,inputsize,hiddensize);
			//buildNormalDist(W2,hiddensize,outputsize);
			float scale1=sqrt(1.0/float(inputsize));
			float scale2=sqrt(1.0/float(hiddensize));
			W1*=scale1;
			W2*=scale2;
			layers[0]=new AffineLayer(W1,b1);
			layers[1]=new SigmoidLayer();
			layers[2]=new AffineLayer(W2,b2);
			lastLayer=new SoftmaxWithLossLayer();
			//layers[3]=lastLayer;
		}
		~TwoLayerNet()
		{
			delete layers[0];
			delete layers[1];
			delete layers[2];
			//delete layers[3];
			delete lastLayer;
		}
		MatrixXd Predict(MatrixXd x)
		{
			for(int i=0;i<3;++i)
			{
				x=layers[i]->forward(x);
			}
			return x;
		}
		MatrixXd Loss(MatrixXd x,MatrixXd t)
		{
			MatrixXd y=Predict(x);
			return ((SoftmaxWithLossLayer*)lastLayer)->forward(y,t);
		}
		void Gradient(MatrixXd x,MatrixXd t)
		{
			Loss(x,t);
			MatrixXd dout=MatrixXd::Constant(1,1,1);
			dout=lastLayer->backward(dout);
			for(int i=0;i<3;++i)
			{
				dout=layers[2-i]->backward(dout);
			}
			gradW1=(((AffineLayer*)layers[0])->dW);
			gradb1=(((AffineLayer*)layers[0])->db);
			gradW2=(((AffineLayer*)layers[2])->dW);
			gradb2=(((AffineLayer*)layers[2])->db);
		}
		void updateLayers()
		{
			((AffineLayer*)layers[0])->W=W1;
			((AffineLayer*)layers[0])->b=b1;
			((AffineLayer*)layers[2])->W=W2;
			((AffineLayer*)layers[2])->b=b2;
		}
		MatrixXd getGradW1();
		MatrixXd getGradW2();
		MatrixXd getGradb1();
		MatrixXd getGradb2();
	};
	
	MatrixXd TwoLayerNet::getGradW1(){return gradW1;}
	MatrixXd TwoLayerNet::getGradW2(){return gradW2;}
	MatrixXd TwoLayerNet::getGradb1(){return gradb1;}
	MatrixXd TwoLayerNet::getGradb2(){return gradb2;}

	/*
	표준정규분포 만들기
	*/
void buildNormalDist(MatrixXd dest,int sx,int sy)
{
	float * nums=new float[sx*sy];
	double sq2pi=sqrt(2*M_PI);
	sq2pi=1.0/sq2pi;
	float x=-10;
	float dx=20.0/(sx*sy);
	for(int i=0;i<sx*sy;++i)
	{
		nums[i]=sq2pi*exp(-x*x/2);
		x+=dx;
	}
	int nDest,nSour;
	float nTemp;
	srand(time(NULL));
	for(int i=0;i<sx*sy*2;i++)
	{
		nDest = rand()%(sx*sy);
		nSour = rand()%(sx*sy);

		nTemp = nums[nDest];
		nums[nDest] = nums[nSour];
		nums[nSour] = nTemp;
	}
	for(int r=0;r<sy;++r)
	{
		for(int c=0;c<sx;++c)
		{
			dest(r,c)=nums[r*sx+c];
		}
	}
	delete[] nums;
}
#define HIDDEN_LAYER 200
TwoLayerNet net(51,HIDDEN_LAYER,/*NUM_Diseases*/31);
	
void LoadWeights(const char *filename)
{
	FILE * input=fopen(filename,"rt");
	if(input==NULL)
	{
		return;
	}
	float data;
	int row,col;
	fscanf(input,"%d %d\n",&row,&col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			net.W1(i,j)=data;
		}
	}
	
	fscanf(input,"%d %d\n",&row,&col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.W2(i,j)=data;
		}
	}/*
	fscanf(input,"%d %d\n",&row,&col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			net.W3(i,j)=data;
		}
	}
	fscanf(input,"%d %d\n",&row,&col);
	
	//printf("W2 %d %d",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.W4(i,j)=data;
		}
	}
	*/
	fscanf(input,"%d %d\n",&row,&col);
	//printf("b1 %d %d",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.b1(i,j)=data;
		}
	}
	
	fscanf(input,"%d %d",&row,&col);
	
	//printf("b2 %d %d",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.b2(i,j)=data;
		}
	}
	/*
	fscanf(input,"%d %d\n",&row,&col);
	//printf("b1 %d %d",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.b3(i,j)=data;
		}
	}
	fscanf(input,"%d %d",&row,&col);
	
	//printf("b2 %d %d",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.b4(i,j)=data;
		}
	}
	*/
	fclose(input);
	net.updateLayers();
}
extern "C"
{
	//JNIEXPORT jstring JNICALL Java_com_kyunggi_medinology_MainActivity_stringFromJNI(JNIEnv* env, jobject thiz)
	//{
	//	return env->NewStringUTF("Hello from JNI !");
	enum GENDER
	{
		MALE,
		FEMALE
	} gender;
	bool preg;
	int age;
	int weight;
	int symptomlen;
	float *symptoms;
	int disease1,disease2,disease3;
	int prob1,prob2,prob3;
	int diseasenum;
	JNIEXPORT void JNICALL Java_com_kyunggi_medinology_MainActivity_initData(JNIEnv* env, jobject thiz,jboolean male, jboolean _preg,jint _age,jint _weight,jbyteArray _symptoms,jint _diseases)
	{
		gender=male==JNI_TRUE?MALE:FEMALE;
		preg=_preg;
		age=_age;
		diseasenum=_diseases;
		symptomlen=env->GetArrayLength(_symptoms);
		symptoms= new float[symptomlen];
		jbyte *byte_buf;
        byte_buf = env->GetByteArrayElements(_symptoms, NULL);
		for(int i=0;i<symptomlen;++i)
		{
			symptoms[i]=(float)byte_buf[i];
		}
		env->ReleaseByteArrayElements(_symptoms, byte_buf, 0);
		//net= TwoLayerNet(72,HIDDEN_LAYER,/*NUM_Diseases*/31);
	}
	JNIEXPORT void JNICALL Java_com_kyunggi_medinology_MainActivity_calcData(JNIEnv* env, jobject thiz)
	{
		MatrixXd x(1,symptomlen);
		for(int i=0;i<symptomlen;++i)
		{
			x(0,i)=symptoms[i];
		}
		MatrixXd result=net.Predict(x);	//1*numsisease
		//result=Softmax(result);
		int i,j;
		prob1=result.maxCoeff(&i,&j)*100;
		result(i,j)=0;
		disease1=j;
		prob2=result.maxCoeff(&i,&j)*100;
		result(i,j)=0;
		disease2=j;
		prob3=result.maxCoeff(&i,&j)*100;
		disease3=j;
	}
	JNIEXPORT jint JNICALL Java_com_kyunggi_medinology_MainActivity_getDisID(JNIEnv* env, jobject thiz,jint n)
	{
		switch(n)
		{
			case 0:
				return disease1;
				break;
			case 1:
				return disease2;
				break;
			case 2:
				return disease3;
				break;
			default:
				break;
		}
		return 0;
	}
	JNIEXPORT jint JNICALL Java_com_kyunggi_medinology_MainActivity_getProb(JNIEnv* env, jobject thiz,jint n)
	{
		switch(n)
		{
			case 0:
				return prob1;
				break;
			case 1:
				return prob2;
				break;
			case 2:
				return prob3;
				break;
			default:
				break;
		}
		return 0;
	}
	
	//////////Unimplemented!!!!!!!!////////
	/////Moved up to java level////////
	/*JNIEXPORT jintArray JNICALL Java_com_kyunggi_medinology_MainActivity_getDrugID(JNIEnv* env, jobject thiz,jint n)
	{
		jintArray ret= env->NewIntArray(3);
		if(ret==NULL)
		{
			return NULL;
		}
		 jint fill[3];
		 int i;
		 int size=3;
		 fill[0]=3;
		 fill[1]=5;
		 fill[2]=6;
		// for (i = 0; i < size; i++) {
	  	//   fill[i] = 0; // put whatever logic you want to populate the values here.
		// }
		 // move from the temp structure to the java structure
	 	env->SetIntArrayRegion(ret, 0, size, fill);
		return ret;
	}
	**/
	
	JNIEXPORT void JNICALL Java_com_kyunggi_medinology_MainActivity_initWeights(JNIEnv* env, jobject thiz/*jstring filename*/)
	{
		//const char *fname = env->GetStringUTFChars( filename, NULL);//Java String to C Style string
		LoadWeights("/sdcard/weight.txt");
 		//env->ReleaseStringUTFChars( filename, );
		logger<<"weight successfully loaded"<<endl;
	}
	

	JNIEXPORT void JNICALL Java_com_kyunggi_medinology_MainActivity_finalizeNative(JNIEnv* env, jobject thiz)
	{
		delete []symptoms;
		//delete net;
	}
}
/*
class TwoLayerNet
{
	public:
		TwoLayerNet(int inputsiz,int hiddensiz,int outputsiz);
		MatrixXd Predict(MatrixXd x);
		MatrixXd W1,W2,W3,W4,b1,b2,b3,b4;
		void CalcGrad(MatrixXd x,MatrixXd t);
		MatrixXd getGradW1();
		MatrixXd getGradW2();
		MatrixXd getGradb1();
		MatrixXd getGradb2();
		MatrixXd getGradW3();
		MatrixXd getGradW4();
		MatrixXd getGradb3();
		MatrixXd getGradb4();	
	private:
		MatrixXd gradW1;
		MatrixXd gradW2;
		MatrixXd gradb1;
		MatrixXd gradb2;
		MatrixXd gradW3;
		MatrixXd gradW4;
		MatrixXd gradb3;
		MatrixXd gradb4;
		
		MatrixXd getGrad(MatrixXd &subject,MatrixXd &x,MatrixXd &t);
		
};

TwoLayerNet::TwoLayerNet(int inputsiz,int hiddensiz,int outputsiz)
{
	float scale=sqrt(1.0/float(inputsiz));
	W1=MatrixXd::Random(inputsiz,outputsiz);
	W1*=scale;
	//W2=MatrixXd::Random(hiddensiz,hiddensiz);
	//W3=MatrixXd::Random(hiddensiz,hiddensiz);
	//W4=MatrixXd::Random(hiddensiz,outputsiz);
	b1=MatrixXd::Zero(1,outputsiz);
	//b2=MatrixXd::Zero(1,hiddensiz);
	//b3=MatrixXd::Zero(1,hiddensiz);
	//b4=MatrixXd::Zero(1,outputsiz);
	
}

MatrixXd TwoLayerNet::getGradW1(){return gradW1;}
MatrixXd TwoLayerNet::getGradW2(){return gradW2;}
MatrixXd TwoLayerNet::getGradW3(){return gradW3;}
MatrixXd TwoLayerNet::getGradW4(){return gradW4;}

MatrixXd TwoLayerNet::getGradb1(){return gradb1;}
MatrixXd TwoLayerNet::getGradb2(){return gradb2;}
MatrixXd TwoLayerNet::getGradb3(){return gradb3;}
MatrixXd TwoLayerNet::getGradb4(){return gradb4;}

MatrixXd TwoLayerNet::Predict(MatrixXd x)
{
	MatrixXd a=x*W1;//+b1;
	//a=Sigmoid(a);
	//a=a*W2+b2;
	//a=Sigmoid(a);
	//a=a*W3+b3;
	//a=Sigmoid(a);
	//a=a*W4+b4;
	a=Softmax(a);
	return a;
}
*/
/*
void TwoLayerNet::CalcGrad(MatrixXd x,MatrixXd t)
{
	//# forward
       MatrixXd a1 = x* W1 + b1;
       MatrixXd z1 = Sigmoid(a1);
       MatrixXd a2 = z1* W2 + b2;
       MatrixXd y = Softmax(a2);
        
       // # backward
        MatrixXd dy = (y - t) / batch_num;
        gradW2= z1.transpose()* dy;
        gradb2 = np.sum(dy, axis=0)
        
        da1 = dy* W2.transpose();
        dz1 = sigmoid_grad(a1) * da1
        gradW1 = x.transpose()* dz1;
        gradsb1 = np.sum(dz1, axis=0);
}

*/
/*
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):	 우리는 batch 하지 않는다
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

	# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

*/
