% --������������ �������ã������ʼ�������ݼ�׼���Ȳ���---------
function [net, info] = Indian_cnn(varargin)
%----CNN_MNIST  Demonstrates MatConvNet on MNIST
%--����..\..\matlab\vl_setupnn.m�����뵱ǰworkspace-----------
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;%�Ƿ�����batchNormģ��
opts.network = [] ;%ѡ�����繹��simplnn��dagnn
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
%����ʵ����·����ţ�data\minst-baseline-simplenn
opts.expDir = fullfile(vl_rootnn, 'data', 'mnist-out', ['Botswana-nothing-' sfx]) ;%vl_rootnn
[opts, varargin] = vl_argparse(opts, varargin) ;
%Mnistԭʼ���ݼ��Ĵ��·����E��\matconvnet-1.0-beta18\data\minst
opts.dataDir = fullfile(vl_rootnn, 'data', 'Indian') ; %û����
%imdb�ṹ��Ĵ��·����data\minst-baseline-simplenn\imdb.mat
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
%�Ƿ�ʹ��GPU�����ʹ�ã���opts.train.gpus=[1];Ϊ����ʹ��
if ~isfield(opts.train, 'gpus'), opts.train.gpus = 1; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
%����ǰ��ָ���Ĳ�����ʼ������
if isempty(opts.network)
  net = Indian_cnn_init('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;
else
  net = opts.network ;
  opts.network = [] ;
end
%��Mnistԭʼ������ȡImdb�ṹ�壬������ھ�ֱ�Ӽ��أ��������getMinstImdb����
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;%����ʵ��Ŀ¼
  save(opts.imdbPath, '-struct', 'imdb') ;%����imdb�ṹ��
end
%����arrayfun��sprintf����Ӧ�õ�����[1:10]��ÿ��Ԫ�أ������ֱ�ǩתΪchar������
net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:16,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
%����ǰ����ѡ�������ṹ��ѡ���Ӧ��ѵ���������洢�ĺ������trainfn��
switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end
%����ѵ����������ʼѵ�����磺find(imdb.image.set==3)�����ҵ���֤������������
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% --����opts��ָ��������ṹ�ͷ���һ��������������ڴ�imdb�ṹ����ȡ������-----
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --����SimpleNN���繹�͵�����������ȡ����������batchΪ��������-------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --����DagNN���繹�͵�����������ȡ����������batchΪ��������-----------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --��MNIST���ݼ��л�ȡ���ݣ���ȥͼ���ֵ����ŵ�Imdb�ṹ����---------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
[x1,y1,x2,y2]=IndiaToMnist_Datapreprocessing(0,5/6);

%���ڱ�ʶ���������е�ѵ�����Ͳ��Լ��ļ��ϣ�set==1���Ӧ������ͼ��ͱ�ǩ������ѵ����==3�����ڲ���
set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];%numel������������������Ԫ���ܺͣ��ȼ���prod��size��A����
data = single(reshape(cat(3, x1, x2),14,14,1,[]));%��x1�е�ѵ������x2�Ĳ��Լ��ڵ���άƴ�������������������ݼ�����unit8��Ϊsingle
dataMean = mean(data(:,:,:,set == 1), 4);%��� ѵ����������ͼ��ľ�ֵͼ��
data = bsxfun(@minus, data, dataMean) ;%bsxfun������minus����Ӧ�õ�data��ÿ��Ԫ���ϣ�������ͼ���ȥ��ֵ
%���imdb�ṹ�幹�����ݼ�
imdb.images.data = data; %size��data��=��28��28��1,70000��;
imdb.images.data_mean = dataMean;%size(dataMean)=[28,28]
imdb.images.labels = cat(2, y1, y2) ;%��ѵ�����Ͳ��Լ��ı�ǩҲƴ��������size��imdb.images.labels��=[1,70000]
imdb.images.set = set ;%size(set)=[1,70000],unique(set)=[1,3]
imdb.meta.sets = {'train', 'val', 'test'} ;%imdb.images.set==1����ѵ����==2������֤��==3���ڲ���
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:16,'uniformoutput',false) ;
