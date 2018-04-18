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
#include "/storage/emulated/0/AppProjects/Medinology2/jni/eigen/Eigen/Dense"
#include <iostream>
#include<stdio.h>
using namespace Eigen;
using namespace std;


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
class TwoLayerNet
{
	public:
		TwoLayerNet(int inputsiz,int hiddensiz,int outputsiz);
		MatrixXd Predict(MatrixXd x);
		MatrixXd W1,W2,b1,b2;
		void CalcGrad(MatrixXd x,MatrixXd t);
		MatrixXd getGradW1();
		MatrixXd getGradW2();
		MatrixXd getGradb1();
		MatrixXd getGradb2();
	private:
		MatrixXd gradW1;
		MatrixXd gradW2;
		MatrixXd gradb1;
		MatrixXd gradb2;	
};
void LoadWeights(const char *filename);

extern "C"
{
	//JNIEXPORT jstring JNICALL Java_com_kyunggi_medinology_MainActivity_stringFromJNI(JNIEnv* env, jobject thiz)
	//{
	//	return env->NewStringUTF("Hello from JNI !");
	#define HIDDEN_LAYER 20
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
	TwoLayerNet net(51,HIDDEN_LAYER,/*NUM_Diseases*/31);
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
	}
	

	JNIEXPORT void JNICALL Java_com_kyunggi_medinology_MainActivity_finalizeNative(JNIEnv* env, jobject thiz)
	{
		delete []symptoms;
		//delete net;
	}
}



TwoLayerNet::TwoLayerNet(int inputsiz,int hiddensiz,int outputsiz)
{
	W1=MatrixXd::Random(inputsiz,hiddensiz);
	W2=MatrixXd::Random(hiddensiz,outputsiz);
	b1=MatrixXd::Random(1,hiddensiz);
	b2=MatrixXd::Random(1,outputsiz);
}
MatrixXd TwoLayerNet::getGradW1(){return gradW1;}
MatrixXd TwoLayerNet::getGradW2(){return gradW2;}
MatrixXd TwoLayerNet::getGradb1(){return gradb1;}
MatrixXd TwoLayerNet::getGradb2(){return gradb2;}

MatrixXd TwoLayerNet::Predict(MatrixXd x)
{
	MatrixXd a=x*W1+b1;
	a=Sigmoid(a);
	a=a*W2+b2;
	a=Softmax(a);
	return a;
}
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
	//printf("W1 %d %d\n",row,col);
	//MessageBox(0,"W1","",0);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			fscanf(input,"%f",&data);
			//printf("%d %d %f\n",i,j,data);
			net.W1(i,j)=data;
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
			net.W2(i,j)=data;
		}
	}
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
	fclose(input);
}
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

