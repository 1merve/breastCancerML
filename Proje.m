%% VERİ ÖN İŞLEME
%veriyi Excel'den içeri aktarma
fileName1 = 'cancer_data.csv'; 
dataTable = readtable(fileName1);

fileName2 = 'cancer_data_y.csv';
targetTable = readtable(fileName2);

% özelliklerin X matrisine atılması
X = table2array(dataTable);
% sınıfın Y'ye atılması
Y= table2array(targetTable);

% eksik verinin olup olmadığının kontrolü
eksikVarMi = any(isnan(X(:)));


%% ÖZELLİK SEÇİMİ
% Min-Max Normalizasyonu

min_deger = min(X);  % Her sütunun minimum değerleri
max_deger = max(X);  % Her sütunun maksimum değerleri
normalMatris = (X - min_deger) ./ (max_deger - min_deger); % Normalleştirilmiş matris

X= normalMatris;


% Normalize edilmiş veri setine özellik seçiminin uygulanması (Korelasyon
% Tabanlı Özellik Seçimi)

% Özelliklerin korelasyon matrisi
corrmatrix = corr(X);
% Hedef değişken ile olan korelasyon
corr_with_target = abs(corr(X, Y));

% Korelasyon tabanlı özellik seçimi
threshold = 0.5; % Korelasyon eşiği
selected_features = find(corr_with_target > threshold);

% Seçilen özelliklerin indekslerini ve korelasyon değerlerini gösterin
disp("Seçilen Özelliklerin İndeksleri:");
disp(selected_features);

disp("Korelasyon Değerleri:");
disp(corr_with_target(selected_features));

% Özellik seçimi aşamasında özelliklerin elenmesinden sonra X veri seti
X = X(:, [1 2 3 4 6 7 8 11 13 14 21 22 23 24 26 27 28]);


%% Modelleme Adımı

% veri setinin eğitim ve test verileri olarak modellenmesi
cvp = cvpartition(Y, 'HoldOut', 0.2); % Her 10 taneden 2'si
dataTrain = X(cvp.training, :);
dataTest = X(cvp.test, :);
YTrain = Y(cvp.training);
YTest = Y(cvp.test);


% K-Nearest Neighbors

% KNN modeli
k = 5; % K değeri (komşu sayısı)
KNNModel = fitcknn(dataTrain, YTrain, 'NumNeighbors', k);

% Eğitim sonucunda test verisinden elde edilen hedef değerleri
result = predict(KNNModel, dataTest);

display(YTest');
display(result');

ccKNN = confusionchart(YTest, result);


% Destek Vektör Makineleri (SVM)

% SVM modeli
SVMModel = fitcsvm(dataTrain, YTrain);

% Eğitim sonucunda test verisinden elde edilen hedef değerleri
SVM_prediction = predict(SVMModel, dataTest);

display(YTest');
display(SVM_prediction');

ccSVM = confusionchart(YTest, SVM_prediction);


% Karar Ağaçları

% Karar ağacı modeli
DTModel = fitctree(dataTrain, YTrain);

% Eğitim sonucunda test verisinden elde edilen hedef değerleri
DT_prediction = predict(DTModel, dataTest);

display(YTest');
display(DT_prediction');

ccDT = confusionchart(YTest, DT_prediction);


% Yapay sinir ağları
% FFBP (feedforward backpropagation)

[~, loc] = ismember(Y, unique(Y));
y_one_hot = ind2vec(loc')';
net = patternnet([20 20]); % 20 + 20 düğümlü hidden layers
net = train(net, X', y_one_hot');
ANN_result = net(X');
cp = classperf(Y, vec2ind(ANN_result));

%% MODEL DEĞERLENDİRME ADIMI

% doğruluk
acc_KNN = sum(YTest == result) / numel(YTest); % KNN için
acc_SVM = sum(YTest == SVM_prediction) / numel(YTest); % Destek Vektör Makineleri için
acc_DT = sum(YTest == DT_prediction) / numel(YTest); % Karar Ağaçları için

display(acc_KNN);
display(acc_SVM);
display(acc_DT);

% hassasiyet
true_positives_KNN = sum(result == 0 & YTest == 0);
false_positives_KNN = sum(result == 0 & YTest == 1);

prec_KNN = true_positives_KNN / (true_positives_KNN + false_positives_KNN);

true_positives_SVM = sum(SVM_prediction == 0 & YTest == 0);
false_positives_SVM = sum(SVM_prediction == 0 & YTest == 1);

prec_SVM = true_positives_SVM / (true_positives_SVM + false_positives_SVM);

true_positives_DT = sum(DT_prediction == 0 & YTest == 0);
false_negatives_DT = sum(DT_prediction == 0 & YTest == 1);

prec_DT = true_positives_DT / (true_positives_DT + false_negatives_DT);

display(prec_KNN);
display(prec_SVM);
display(prec_DT);

% duyarlılık = TP / (TP + FN)
true_positives_KNN = sum(result == 0 & YTest == 0);
false_negatives_KNN = sum(result == 1 & YTest == 0);

recall_KNN = true_positives_KNN / (true_positives_KNN + false_negatives_KNN);

true_positives_SVM = sum(SVM_prediction == 0 & YTest == 0);
false_negatives_SVM = sum(SVM_prediction == 1 & YTest == 0);

recall_SVM = true_positives_SVM / (true_positives_SVM + false_negatives_SVM);

true_positives_DT = sum(DT_prediction == 0 & YTest == 0);
false_negatives_DT = sum(DT_prediction == 1 & YTest == 0);

recall_DT = true_positives_DT / (true_positives_DT + false_negatives_DT);

display(recall_KNN);
display(recall_SVM);
display(recall_DT);
