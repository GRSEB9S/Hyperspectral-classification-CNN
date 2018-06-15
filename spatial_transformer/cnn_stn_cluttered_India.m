function [net, info] = cnn_stn_cluttered_India(varargin)
%CNN_STN_CLUTTERED_MNIST Demonstrates training a spatial transformer
%   The spatial transformer network (STN) is trained on the 
%   cluttered MNIST dataset.

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(vl_rootnn, 'data') ;
opts.useSpatialTransformer = true ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataPath = fullfile(opts.dataDir,'cluttered-India.mat')  ;
if opts.useSpatialTransformer
  opts.expDir = fullfile(vl_rootnn, 'data', 'cluttered-India-stn') ;
else
  opts.expDir = fullfile(vl_rootnn, 'data', 'cluttered-India-no-stn') ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.dataURL = 'http://www.vlfeat.org/matconvnet/download/data/cluttered-mnist.mat' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdDB(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
net = cnn_stn_cluttered_India_init([14 14], true) ; % initialize the network
net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:16,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

fbatch = @(i,b) getBatch(opts.train,i,b);
[net, info] = cnn_train_dag(net, imdb, fbatch, ...
                            'expDir', opts.expDir, ...
                            net.meta.trainOpts, ...
                            opts.train, ...
                            'val', find(imdb.images.set == 2)) ;

% --------------------------------------------------------------------
%                                                     Show transformer
% --------------------------------------------------------------------

figure(100) ; clf ;
v = net.getVarIndex('xST') ;
net.vars(v).precious = true ;
net.eval({'input',imdb.images.data(:,:,:,1:6)}) ;
for t = 1:6
  subplot(2,6,t) ; imagesc(imdb.images.data(:,:,:,t)) ; axis image off ;
  subplot(2,6,6+t) ; imagesc(net.vars(v).value(:,:,:,t)) ; axis image off ;
  colormap gray ;
end

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------

if ~isa(imdb.images.data, 'gpuArray') && numel(opts.gpus) > 0
  imdb.images.data = gpuArray(imdb.images.data);
  imdb.images.labels = gpuArray(imdb.images.labels);
end
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getImdDB(opts)
% --------------------------------------------------------------------

% Prepare the IMDB structure:
% if ~exist(opts.dataDir, 'dir')
%   mkdir(opts.dataDir) ;
% end
% if ~exist(opts.dataPath)
%   fprintf('Downloading %s to %s.\n', opts.dataURL, opts.dataPath) ;
%   urlwrite(opts.dataURL, opts.dataPath) ;
% end
% dat = load(opts.dataPath);
[x_tr,x_vl,x_ts,y_tr,y_vl,y_ts] = STN_India_Datapreprocessing([3 1 1]);

set = [ones(1,numel(y_tr)) 2*ones(1,numel(y_vl)) 3*ones(1,numel(y_ts))];
data = single(cat(4,x_tr,x_vl,x_ts));
imdb.images.data = data ;
imdb.images.labels = single(cat(2, y_tr,y_vl,y_ts)) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:16,'uniformoutput',false) ;
