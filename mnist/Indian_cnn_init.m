function net = Indian_cnn_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = false ;%Ĭ�ϴ�batchNormalization
opts.networkType = 'simplenn' ;%Ĭ��ʹ��Simplenn����ṹ
opts = vl_argparse(opts, varargin) ;%�����ⲿ�����޸�Ĭ��ֵ��

rng('default');%�����������������ʹ��ÿ�����н��������
rng(0) ;
%����ṹ��conv->maxpool->conv->maxpool->conv->relu->conv->softmaxloss
f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...%randn(h,w,d,n)����4D��׼��̬�ֲ�������ΪMNIST���ݼ���ͼ���ǵ�ͨ���ģ���Ԫͻ��Ϊ1����d=1
                           'weights', {{f*randn(3,3,1,20, 'single'), zeros(1, 20, 'single')}}, ...%��Ԫ����Ϊ20����ƫ��20
                           'stride', 1, ...%�˲����Ļ�������Ϊ1��5*7
                           'pad', 0) ;%ͼ��ı߽���չΪ0
%net.layers{end+1} = struct('type', 'dropout') ;%RelU��
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...%����ػ���ʽ
                           'pool', [2 2], ...%2*2,4������ȡ���
                           'stride', 2, ...%��������Ϊ2.
                           'pad', 0) ;%�߽�����չ
net.layers{end+1} = struct('type', 'conv', ...%��Ԫ��Ŀ50������Ϊ�ϲ���20��Ԫ�����Ա���ÿ����Ԫ20��ͻ��20��
                           'weights', {{f*randn(3,3,20,50, 'single'),zeros(1,50,'single')}}, ...%ƫ��������������Ԫ������ȣ�����50����
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [1 1], ...
                           'stride', 1, ...
                           'pad', 0) ;       
%net.layers{end+1} = struct('type', 'dropout') ;%RelU��                       
net.layers{end+1} = struct('type', 'conv', ...%��Ԫ��Ŀ500����Ϊ�ϲ�50����Ԫ�����Ա���û����Ԫͻ��50��
                           'weights', {{f*randn(3,3,50,500, 'single'),  zeros(1,500,'single')}}, ...%ƫ����Ŀ500��
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;%RelU��
%net.layers{end+1} = struct('type', 'dropout') ;%RelU��                       
net.layers{end+1} = struct('type', 'conv', ...%��Ԫ��Ŀ10������MNIST�ܹ�10�ࣩ�ϲ�500��Ԫ�����Ա���ͻ��500��
                           'weights', {{f*randn(1,1,500,16, 'single'), zeros(1,16,'single')}}, ...%ƫ������10��������˵ĳߴ�Ϊ1��ȫ����
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;%�������ʧ�������softmax��logistic��ʧ����ʹ���һ������໥����
%net.layers{end+1} = struct('type', 'loss') ;%�������ʧ�������softmax��logistic��ʧ����ʹ���һ������໥����


% optionally switch to batch normalization���ݲ���ѡ�����bath normalization��
if opts.batchNormalization
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  net = insertBnorm(net, 7) ;
  %���opts.batchNormalizationΪ�棬�����ԭ����nets�ϲ������㣬net�ṹ��Ϊ��
  % conv->bnorm->maxpool->conv->maxpool->bnorm->conv->relu->conv->bnorm->softmaxloss
end

% Meta parameters����ṹԪ����
net.meta.inputSize = [14 14 1] ;%�������ݳߴ磺28*28�ĵ�ͨ��
net.meta.trainOpts.learningRate = [0.0005*ones(1,30) 0.0001*ones(1,10) 0.00001*ones(1,10)] ;%ѧϰ��
net.meta.trainOpts.numEpochs = 50;%ѵ���غϴ���
net.meta.trainOpts.batchSize = 20 ;%һ�����εĴ�����������

% Fill in defaul values�������Ĭ������ֵ
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested�������Ҫ��ת��ΪDagNNĬ��ʹ��Simplenn����ṹ
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
      'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% ----------------------��net�ĵ�L����L+1�����bnorm��---------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));%ȷ����L����Ȩ����
ndim = size(net.layers{l}.weights{1}, 4);%ȡ�õ�L�����Ԫ����
%��ʼ��һ��Bnorm��Layer��
layer = struct('type', 'lrn', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...%bnormȨ����������һ����Ԫ������ͬ
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
%���´�����Bnorm layer�������net�ĵ�L�����L+1���м�
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
