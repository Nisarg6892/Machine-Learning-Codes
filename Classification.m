function proj3 = proj3

% function images = loadMNISTImages(filename)
% %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
% %the raw MNIST images
% fp_images = fopen(filename_images, 'rb');
% assert(fp_images ~= -1, ['Could not open ', filename_images, '']);
% magic_images = fread(fp_images, 1, 'int32', 0, 'ieee-be');
% assert(magic_images == 2051, ['Bad magic number in ', filename_images, '']);
% numImages = fread(fp_images, 1, 'int32', 0, 'ieee-be');
% numRows_images = fread(fp_images, 1, 'int32', 0, 'ieee-be');
% numCols_images = fread(fp_images, 1, 'int32', 0, 'ieee-be');
% images = fread(fp_images, inf, 'unsigned char');
% images = reshape(images, numCols_images, numRows_images, numImages);
% % images = permute(images,[2 1 3]);
% fclose(fp_images);
% % Reshape to #pixels x #examples
% images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% % Convert to double and rescale to [0,1]
% images = double(images) / 255;
% end
% 
% 
% function images_train = loadMNISTImages(filename)
% %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
% %the raw MNIST images
% fp_images_train = fopen(filename_images_train, 'rb');
% assert(fp_images_train ~= -1, ['Could not open ', filename_images_train, '']);
% magic_images_train = fread(fp_images_train, 1, 'int32', 0, 'ieee-be');
% assert(magic_images_train == 2051, ['Bad magic number in ', filename_images_train, '']);
% numImages_train = fread(fp_images_train, 1, 'int32', 0, 'ieee-be');
% numRows_train = fread(fp_images_train, 1, 'int32', 0, 'ieee-be');
% numCols_train = fread(fp_images_train, 1, 'int32', 0, 'ieee-be');
% images_train = fread(fp_images_train, inf, 'unsigned char');
% images_train = reshape(images_train, numCols_train, numRows_train, numImages_train);
% % images_train = permute(images_train,[2 1 3]);
% fclose(fp_images_train);
% % Reshape to #pixels x #examples
% images_train = reshape(images_train, size(images_train, 1) * size(images_train, 2), size(images_train, 3));
% % Convert to double and rescale to [0,1]
% images_train = double(images_train) / 255;
% end

x_dimension = 10;
x0 = zeros(length(labels_train),x_dimension);
x1 = zeros(length(labels),x_dimension);

for i=1:length(labels_train)
    x0(i,(labels_train(i))+1) = 1; 
end

for i=1:length(labels)
    x1(i,labels(i)+1) = 1; 
end

FinalImages_train_output = ([ones(1,length(images_train));images_train(:,:)]);
FinalImages_test_output = ([ones(1,length(images));images(:,:)]);
Layer2_Nodes = length(FinalImages_train_output(:,1));
Layer3_Nodes = x_dimension;
weightVectorInitialLayer_1 = randn(length(FinalImages_train_output(:,1)),Layer2_Nodes); 
weightVectorInitialLayer_2 = randn(Layer2_Nodes+1,Layer3_Nodes);
weightVectorLayer_1 = weightVectorInitialLayer_1;
weightVectorLayer_2 = weightVectorInitialLayer_2;
eta1 = 0.06;
eta2 = 0.06;

for i= 1:length(FinalImages_train_output)
   imageSample = FinalImages_train_output(:,i);
   outputAtLayer_2 = sigmoid(transpose(weightVectorLayer_1) * imageSample,0.01,0);
   inputToLayer_3 = [1;outputAtLayer_2(:,:)];
   finalOutput = sigmoid(transpose(weightVectorLayer_2) * inputToLayer_3,0.01,0);
   deltaLayer_3 = finalOutput - transpose(x0(i,:));
   deltaLayer_2 = outputAtLayer_2 .* (1- outputAtLayer_2) .* (weightVectorLayer_2(2:length(weightVectorLayer_2),:) * deltaLayer_3);
   weightVectorLayer_1 = weightVectorLayer_1 - transpose(eta1 * deltaLayer_2 * transpose(imageSample));
   weightVectorLayer_2 = weightVectorLayer_2 - transpose(eta2 * deltaLayer_3 * transpose(inputToLayer_3));         
end

count = 0;
for i=1:length(FinalImages_test_output)
    imageSample = FinalImages_test_output(:,i);
    outputAtLayer_2 = sigmoid(transpose(weightVectorLayer_1) * imageSample,0.01,0);
    inputToLayer_3 = [1;outputAtLayer_2(:,:)];
    finalOutput = sigmoid(transpose(weightVectorLayer_2) * inputToLayer_3,0.01,0);
    [M,I]= max(abs(finalOutput));
    if M < 0.5
        fprintf('max Value: %d\t',M);
    end
    actualResult = transpose(x1(i,:));
    if actualResult(I)== 1
        count = count +1;
        fprintf('count: %d \n',count);
    end  
end
Wnn1 = weightVectorLayer_1(2:length(weightVectorLayer_1(:,1)),:);
bnn1 = weightVectorLayer_1(1,:);
Wnn2 = weightVectorLayer_2(2:length(weightVectorLayer_2(:,1)),:);
bnn2 = weightVectorLayer_2(1,:);

y_dimension = 10;
y_train = zeros(length(labels_train),y_dimension);
y_test = zeros(length(labels),y_dimension);
for i=1:length(labels_train)
    y_train(i,(labels_train(i))+1) = 1; 
end

for i=1:length(labels)
    y_test(i,labels(i)+1) = 1; 
end

w0 = randn(785,10);
w1 = w0;
FinalImages_train_output = ([ones(1,length(images_train));images_train(:,:)]);
FinalImages_test_output = ([ones(1,length(images));images(:,:)]);
eta =0.12;

for i= 1:length(images_train)
    w1 = w1 - eta * transpose((customSoftmax(transpose(w1)*FinalImages_train_output(:,i))-transpose(y_train(i,:))) * transpose(FinalImages_train_output(:,i)));
end


%%%% testing dataset
finalOutput = zeros(length(y_test),length(y_test(1,:)));
for i= 1:length(finalOutput)
   finalOutput(i,:) = transpose(customSoftmax(transpose(w1)*FinalImages_test_output(:,i)));
end

count = 0;

Wlr = w1(2:length(w1(:,1)),:);
blr = w1(1,:);

function mu = customSoftmax(eta)
mu = exp(eta) ./ sum(exp(eta));

function m = sigmoid(eta,rangeStart,rangeEnd)
% %m = 1 ./ (1+exp(-1 * rangeStart *(eta-rangeEnd)));
m = 1 ./ (1 + exp(-1 * eta));

% Convolution Neural Network
function test_example_CNN
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);

assert(er<0.12, 'Too big error');
