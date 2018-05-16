







#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

double alpha = 0.0000000008;
int kkk = 0;
double xxx = 0;
class node{
private:
    
    double sum = 0;
    double output = 0;
    double delta;
    vector<double> weight;
    vector<node*> connected;
    vector<node*> forwardConnected;

    
public:
    
    double out(){
        return output;
    } // basically for output layer
    void in(double x){
        sum = x;
        output = x;
    }// for input layer
    void setConnect(vector<node*> connect){
        connected = connect;
        
        for (int i=0; i < connect.size(); i++){
            connect[i]->forwardConnected.push_back(this);
        }
        weight.resize(connect.size());
        for (int i=0; i<weight.size(); i++) {
            while (weight[i]<=0.05) {
                weight[i] = (double)rand()/RAND_MAX;
            }
        }
    }
    void feedForward(bool outputLayer){
        sum = 0;
        for (int i=0; i<weight.size(); i++) {
            sum = sum + connected[i]->out()*weight[i];
        }
        output = sum;
    }
    
    void calculateDelta(bool isOutputLayer, double tar){
        if (isOutputLayer) {
            delta = 1;
            xxx = tar - output;
            //xxx = xxx;
        }
        else{
            double tmp = 0;
            for (int i=0; i<forwardConnected.size(); i++) {
                tmp = tmp + forwardConnected[i]->weight[kkk]*forwardConnected[i]->delta;
            }
            kkk++;
            delta = tmp;
        }
    }
    
    void update(){
        for (int i=0; i<weight.size(); i++) {
            weight[i] = weight[i] + alpha*connected[i]->output*delta*xxx;
        }
    }
    
};



class layer {
private:
    
    vector<node> Node;
    
public:
    
    layer(int nodeCount){
        Node.resize(nodeCount+1); // 1 for dummy node
    }
    vector<double> getOutput(){
        vector<double> out;
        for (int i=0; i<Node.size(); i++) {
            out.push_back(Node[i].out());
        }
        return out;
    }//output layer
    
    
    void setInput(vector<double> x){
        for (int i=0; i<x.size(); i++) {
            Node[i].in(x[i]);
        }
    }//inputlayer
    int nodeSize(){
        return Node.size();
    }
    void connectLayer(layer *L){
        vector<node*> n;
        for (int i=0; i<this->Node.size(); i++) {
            n.push_back(&Node[i]);
        }
        for (int i=0; i<L->nodeSize(); i++) {
            L->Node[i].setConnect(n);
        }
    }
    void feedFoward(bool isOutputLayer){
        for (int i=0; i<Node.size()-1; i++) {
            Node[i].feedForward(isOutputLayer);
        }
    }//feed forward
    void backProp(bool isOutputLayer, vector<double> t){
        kkk = 0;
        for (int i=0; i<Node.size()-1; i++) {
            Node[i].calculateDelta(isOutputLayer, t[i]);
        }
    }
    void update(){
        for (int i=0; i<Node.size(); i++) {
            Node[i].update();
        }
    }
    
};




class neuralNet{
private:
    
    vector<layer> L;
    void feedForward(vector<double> x){
        L[0].setInput(x);
        for (int i=1; i<L.size()-1; i++) {
            L[i].feedFoward(false); //false for not output layer
        }
        L[L.size()-1].feedFoward(true);
    }
    void backPropagate(vector<double> tar, vector<double> y){
        for (int i=L.size()-1; i>=0; i--) {
            if (i==L.size()-1) {
                L[i].backProp(true, tar);
            }
            else{
                L[i].backProp(false, tar);//tar here is ok, wont use seconde parameter
            }
        }
    }
    void update(){
        for (int i=1; i<L.size(); i++) {
            L[i].update();
        }
    }
    
public:
    
    void setInput(vector<double> x){
        L[0].setInput(x);
    }
    vector<double> getOutput(){
        return L[L.size()-1].getOutput();
    }
    vector<double> predict(vector<double> x){
        feedForward(x);
        return L[L.size()-1].getOutput();
    }
    void createLayer(int nodeCount){
        L.push_back(layer(nodeCount));
        if (L.size()>=2) {
            L[L.size()-2].connectLayer(&L[L.size()-1]);
        }
    }
    void learnData(vector<double> x, vector<double> tar){
        feedForward(x);
        backPropagate(tar, getOutput());
        update();
    }
    
    
};



int main() {
    
    neuralNet n;
    vector <double> x;
    vector <double> y;
    
    vector <double> test;
    
    
    n.createLayer(1);
    n.createLayer(3);
    n.createLayer(2);
    n.createLayer(3);
    n.createLayer(1);
    
    //testing data creation
    for (int k=0; k<1000000; k++) {
        for (int i=1; i < 100; i++){
            x.clear();
            y.clear();
            x.push_back(i);
            x.push_back(1);//dummy
            y.push_back(2*i*i + 3);
            n.setInput(x);
            n.learnData(x, y);
        }
    }
    
    
    
    
    
    test.push_back(1);
    vector<double> t = n.predict(test);
    printf("%lf\n",t[0]);
    
    
    return 0;
    
}
