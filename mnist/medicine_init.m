function net = medicine_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...%randn(h,w,d,n)����4D��׼��̬�ֲ�������ΪMNIST���ݼ���ͼ���ǵ�ͨ���ģ���Ԫͻ��Ϊ1����d=1
                           'weights', {{f*randn(3,3,1,20, 'single'), zeros(1, 20, 'single')}}, ...%��Ԫ����Ϊ20����ƫ��20
                           'stride', 1, ...%�˲����Ļ�������Ϊ1��5*7
                           'pad', 0) ;%ͼ��ı߽���չΪ0
%net.layers{end+1} = struct('type', 'dropout') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...%����ػ���ʽ
                           'pool', [3 3], ...%2*2,4������ȡ���
                           'stride', 3, ...%��������Ϊ2.
                           'pad', 0) ;%�߽�����չ
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,20,50, 'single'),zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
%net.layers{end+1} = struct('type', 'dropout') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [4 4], ...
                           'stride', 4, ...
                           'pad', 0) ;       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(16,16,50,500, 'single'),  zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'dropout') ;                     
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,500,6, 'single'), zeros(1,6,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.batchNormalization
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  net = insertBnorm(net, 7) ;
end

% Meta parameters
net.meta.inputSize = [200 200 1] ;
net.meta.trainOpts.learningRate = [1e-4*ones(1,50) 5e-7*ones(1,30) 1e-7*ones(1,20)];
net.meta.trainOpts.numEpochs = 100 ;
net.meta.trainOpts.batchSize = 10;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
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

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
