%Program BPNN Iris Clasification 
%Kelompok BISA :
%%Irmansyah Turhamun
%%Jauharil F. Bassam
%%Rizky Nurfaizi
%%Sandyka G.P.
%--------------------------------------------
%Clear Environment and variable before start
clear all;
close all;
clc

%Package io untuk membaca csv (https://www.kaggle.com/c/digit-recognizer/discussion/4667)
%pkg load io
data = csvread('iris_converted.csv',1,0);
for feat = 1:4 %Scaling sehingga range data hanya berkisar 0 sampai 1
  data(:,feat) = (data(:,feat)-min(data(:,feat)))/(max(data(:,feat))-min(data(:,feat)));
end
jumlah_data = max(size(data));

%Definisi jumlah neuron, gambar seperti Gambar 3 di Lecture's Notes
n = 4 %Jumlah input
m = 5 %Jumlah hidden layer 1
l = 3 %jumlah output, mengikuti class iris
X = zeros (n,1); %Node X (input)
Z = zeros (m,1); %Node Z (hidden layer 1)
Y = zeros (l,1); %Node Y (output)
X0 = 0 %Node Bias X (input)
Z0 = 0 %Node Bias Z (hidden layer 1)
alpha = 0.01 %Learning Rate
beta = 0.5 %Momentum, persentase pengaruh pembelajaran sebelumnya

%Memisahkan data untuk Training dan Testing
ratioTT = 6/10 %Rasio Training Set/Total Data (yang sisanya dijadikan Testing Set)
jumlah_dataTraining = floor(ratioTT*jumlah_data);
jumlah_dataTesting = jumlah_data - jumlah_dataTraining;
dataTraining = [];
dataTesting = [];
for u = 1:l
  dataTraining = [dataTraining; data(((u-1)*(jumlah_data/l))+1:(((u-1)*(jumlah_data/l))+(jumlah_data/l)*ratioTT),:)];
  dataTesting = [dataTesting; data((((u-1)*(jumlah_data/l))+(jumlah_data/l)*ratioTT)+1:((u)*(jumlah_data/l)),:)];
  %Memisahkan data Training dan Testing secara rata
end

k = 5 %Nilai k pada K-Fold Validation

V = rand(n,m);                                                                                      %1.Inisialisasi bobot Vnm, dan Wml dan bias V0m dan W0l dengan metode Nguyen-Widrow
W = rand(m,l);
V0 = rand(1,m);
W0 = rand(1,l);
%Belum digunakan metode Nguyen Widrow (metode random)

%Definisi variabel lain
deltaWml = zeros (m,l);
deltaVnm = zeros (n,m);
MSSE = 1 %Inisialisasi MSSE agar tidak error dalam Epoch pertama, nanti dihapus nilai 1 nya
Best_MSSE = 1 %Variabel MSSE terrendah
Epoch = 0
Max_Epoch = 300 %Maksimal pembelajaran

while (Epoch < Max_Epoch)                                                                               %2.While stopping criteria FALSE
  
  dataTraining = dataTraining(randperm(size(dataTraining, 1)), :); 
  %Mengacak isi setiap iterasi K-Fold baru agar tidak berurut sesuai class
  
  for k_iter = 1:k
    
    dataValidation = []; %Inisialisasi data untuk Validation dan reset
  
    Epoch = Epoch + 1
    %Mencatat iterasi

    %Training Phase
    for i = 1:jumlah_dataTraining                                                                        %3.For setiap pasangan data pelatihan dan targetnya 
                                                                                                   %(Feed forward)
    
    %Jika data tersebut adalah dataset Validation, maka disimpan datanya dan skip iterasi
    if ((k_iter-1)*jumlah_dataTraining/k < i && k_iter*jumlah_dataTraining/k >= i)
      dataValidation = [dataValidation;dataTraining(i,:)];
      continue
    end
    
    %for i1 = 1:n                                                                                  %4.For setiap unit input Xn
      %%Ini mungkin maksudnya membaca file nya 1 sel 
      %%(karena realitanya nggak mungkin baca jutaan sel csv dulu sebelum mulai) 
      %%tapi kita sudah load semua karena cuman sedikit barisnya                                   %5.Terima input Xn
      %x(i1) = dataTraining(i,i1)                                                                          %6.Kirim input ke layer di atasnya (hidden layer)
    %end                                                                                        %7.End for
    X = dataTraining(i, 1 : n)'; %Summary dari langkah atas biar singkat

    %Proses dari input ke hidden layer 1  
    for i2 = 1:m                                                                                   %8.For setiap unit hidden Zm
      temp = V0(1,i2) + sum(X.*V(:,i2));                                                            %9.Hitung semua sinyal input dengan bobot dan biasnya sesuai (1.1)
      Z_in(i2) = 1/(1+exp(-temp));                                                                  %10.Hitung nilai aktivasi setiap unit hidden sesuai (1.3) (https://octave.org/doc/v4.0.1/Exponents-and-Logarithms.html)
                                                                                                   %Rumus Sigmoid Function di Lecture's Notes salah (http://mathworld.wolfram.com/SigmoidFunction.html)
      Z(i2,1) = Z_in(i2);                                                                           %11.Kirim nilai aktivasi sebagai input untuk unit output
    end                                                                                         %12.End for

    
    %Proses dari hidden layer 1 ke output
    for i3 = 1:l                                                                                   %13.For setiap unit output Ym
      Y_in(i3) = W0(1,i3) + sum(Z.*W(:,i3));                                                        %14.Hitung  semua  sinyal  input  yang  didapatkan  dari  layer  sebelumnya  dengan  bobot dan biasnya sesuai (1.4)
      Y(i3,1) = 1/(1+exp(-Y_in(i3)));                                                               %15.Hitung nilai aktivasi setiap unit output sebagai output dari jaringan, sesuai (1.5)
    end                                                                                         %16.End for
 
 
 
    %Proses mendeteksi error                                                                                               %(Back propagation of error)
    for i4 = 1:l                                                                                   %17.For setiap unit output Yl 
      harusnyal = 0.10 + (i4-1 == dataTraining(i,5))*0.70;                                                    %18.Terima pola target yang seharusnya Tn
      %Ekuivalen dengan "If 0 maka [0.7 0.15 0.15], if 1 maka [0.15 0.7 0.15], dsb"
      error_l(i4) = ((harusnyal - Y(i4,1)))*((1/(1+exp(-Y_in(i4)))) * (1-(1/(1+exp(-Y_in(i4))))));    %19.Hitung informasi error ?l sesuai (1.6)
                                                                                                     %https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
      deltaWml(:,i4) = alpha*error_l(i4).*Z + beta*deltaWml(:,i4);                                  %20.Hitung besarnya koreksi bobot dari hidden ke output layer, sesuai (1.8)
      deltaW0 = alpha*error_l(i4);                                                                  %21.Hitung besarnya koreksi bias dari hiddenke output layer, sesuai (1.9)
      %Karena diprogram di satu file/function yang sama,
      %maka tidak perlu untuk ditransfer ke layer bawahnya                                         %22.Kirim nilai ??pada layer di bawahnya      
    end                                                                                         %23.End for

    
    
    for i5 = 1:m                                                                                   %24.For setiap unit hiddenZm
      error_m(i5) = sum(error_l.*W(i5,:))*(1/(1+exp(-Z_in(i5))) - (1-(1/(1+exp(-Z_in(i5))))));      %25.Hitung informasi error ??sesuai (1.12)
      %Catatan : Rumus diatas sedikit berbeda dengan Lecture's Notes karena harusnya ada Sigma/SUM pada rumus (1.12)
      %          (Jika tidak maka matrix tidak sesuai) (https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
      deltaVnm(:,i5) = alpha*error_m(i5).*X + beta*deltaVnm(:,i5);                                  %26.Hitung besarnya koreksi bobot dari input ke hidden layer sesuai (1.13)
      deltaV0 = alpha*error_m(i5);                                                                  %27.Hitung besarnya koreksi bias dari input ke hiddenlayer, sesuai (1.14)
    end                                                                                         %28.End for
     
     
    %Update bobot 
    W = W + deltaWml;                                                                               %29.Hitung bobot dan bias dari input ke hidden layer yang baru sesuai (1.10) dan (1.11)
    W0 = W0 + deltaW0; 
    V = V + deltaVnm;                                                                               %30.Hitung  bobot  dan  bias  dari hiddenke output  layeryang  baru  sesuai   (1.15) dan (1.16)
    V0 = V0 + deltaV0;

    end                                                                                           %31.End for
    
  
    %Validation Phase  
    error_validasi = []; %Inisialisasi
    for j = 1:max(size(dataValidation))
    
      X = dataValidation(j, 1 : n)'; %Ambil data ke input  
    
      %Input ke hidden layer 1
      for j2 = 1:m                                                                                   %8.For setiap unit hidden Zm
        temp = V0(1,j2) + sum(X.*V(:,j2));                                                            %9.Hitung semua sinyal input dengan bobot dan biasnya sesuai (1.1)
        Z_in(j2) = 1/(1+exp(-temp));                                                                  %10.Hitung nilai aktivasi setiap unit hidden sesuai (1.3) (https://octave.org/doc/v4.0.1/Exponents-and-Logarithms.html)
                                                                                                     %Rumus Sigmoid Function di Lecture's Notes salah (http://mathworld.wolfram.com/SigmoidFunction.html)
        Z(j2,1) = Z_in(j2);                                                                           %11.Kirim nilai aktivasi sebagai input untuk unit output
      end                                                                                         %12.End for

      
      %hidden layer 1 ke output
      for j3 = 1:l                                                                                   %13.For setiap unit output Ym
        Y_in(j3) = W0(1,j3) + sum(Z.*W(:,j3));                                                        %14.Hitung  semua  sinyal  input  yang  didapatkan  dari  layer  sebelumnya  dengan  bobot dan biasnya sesuai (1.4)
        Y(j3,1) = 1/(1+exp(-Y_in(j3)));                                                               %15.Hitung nilai aktivasi setiap unit output sebagai output dari jaringan, sesuai (1.5)
      end                                                                                         %16.End for
    
      %Melihat hasil seharusnya
      for j4 = 1:l
        harusnyal = 0.10 + (j4-1 == dataValidation(j,5))*0.70;     
        %Ekuivalen dengan "If 0 maka [0.7 0.15 0.15], if 1 maka [0.15 0.7 0.15], dsb"
        error_feature(j4) = (harusnyal - Y(j4,1));
      end
      
      error_validasi = [error_validasi;error_feature]; %Mencatat error setiap feature dan dataset nya
    
    end
    
    %Perhitungan error
    MSSE = [MSSE sqrt(sum((sum((error_validasi).^2))/max(size(error_validasi)))/l)];                                 %32.Hitung error MSSE, sesuai (1.17)
    
    if (Best_MSSE > MSSE(max(size(MSSE)))) %Mencatat variabel dengan MSSE terrendah
      Best_MSSE = MSSE(max(size(MSSE)));
      Best_Epoch = Epoch;
      Best_V = V;
      Best_V0 = V0;
      Best_W = W;
      Best_W0 = W0;
    end
  
    if (Epoch >= Max_Epoch) %Keluar jika sudah mencapai batas Max_Epoch
      %Hanya akan terjadi jika Max_Epoch tidak dapat dibagi dengan nilai k pada K-Fold
      break
    end
  
  end  
    
end                                                                                           %33.End while

    MSSE(1) = []; %Hapus bekas inisialisasi 1
    %Menggunakan V,V0,W,W0 dengan error yang terbaik (paling rendah)
    V = Best_V;
    V0 = Best_V0;
    W = Best_W;
    W0 = Best_W0;   
    
    
    scatter (1:Epoch,MSSE)
    hold on
    plot (1:Epoch,MSSE)
    %Melihat progress di grafik
    
    hold off
    
    %Testing Phase Dataset Training
    %Inisialisasi
    jumlah_sukses_training = zeros(1,3);
    for u = 1:l 
      dataTraining = [dataTraining; data(((u-1)*(jumlah_data/l))+1:(((u-1)*(jumlah_data/l))+(jumlah_data/l)*ratioTT),:)];
      %Mengurutkan kembali dengan cara mengambil dari file aslinya
    end
    
    for h = 1:jumlah_dataTraining
      
      %Ambil data taruh di input
      X = dataTraining(h, 1 : n)';  
    
      %Input ke hidden layer 1
      for h2 = 1:m                                                                                   %8.For setiap unit hidden Zm
        temp = V0(1,h2) + sum(X.*V(:,h2));                                                            %9.Hitung semua sinyal input dengan bobot dan biasnya sesuai (1.1)
        Z_in(h2) = 1/(1+exp(-temp));                                                                  %10.Hitung nilai aktivasi setiap unit hidden sesuai (1.3) (https://octave.org/doc/v4.0.1/Exponents-and-Logarithms.html)
                                                                                                     %Rumus Sigmoid Function di Lecture's Notes salah (http://mathworld.wolfram.com/SigmoidFunction.html)
        Z(h2,1) = Z_in(h2);                                                                           %11.Kirim nilai aktivasi sebagai input untuk unit output
      end                                                                                         %12.End for

      
      %Hidden layer 1 ke output
      for h3 = 1:l                                                                                   %13.For setiap unit output Ym
        Y_in(h3) = W0(1,h3) + sum(Z.*W(:,h3));                                                        %14.Hitung  semua  sinyal  input  yang  didapatkan  dari  layer  sebelumnya  dengan  bobot dan biasnya sesuai (1.4)
        Y(h3,1) = 1/(1+exp(-Y_in(h3)));                                                               %15.Hitung nilai aktivasi setiap unit output sebagai output dari jaringan, sesuai (1.5)
      end                                                                                         %16.End for
      
      %Mencoba menjawab dengan mengambil nilai tertinggi di suatu output
      value = Y(1,1);
      jawaban = 1;
      for h4 = 2:l
        if Y(h4,1) > value
          value = Y(h4,1);
          jawaban = h4;
        end
      end
      
      %Mencocokkan jawaban
      if jawaban == dataTraining(h,5)+1
        jumlah_sukses_training(dataTraining(h,5)+1) = jumlah_sukses_training(dataTraining(h,5)+1) + 1;
      end
    
    end
    
    %Testing Phase Dataset Testing
    %Inisialisasi
    jumlah_sukses_testing = zeros(1,3);
    for h = 1:jumlah_dataTesting
      
      %Ambil data taruh di input
      X = dataTesting(h, 1 : n)';  
    
      %Input ke hidden layer 1
      for h2 = 1:m                                                                                   %8.For setiap unit hidden Zm
        temp = V0(1,h2) + sum(X.*V(:,h2));                                                            %9.Hitung semua sinyal input dengan bobot dan biasnya sesuai (1.1)
        Z_in(h2) = 1/(1+exp(-temp));                                                                  %10.Hitung nilai aktivasi setiap unit hidden sesuai (1.3) (https://octave.org/doc/v4.0.1/Exponents-and-Logarithms.html)
                                                                                                     %Rumus Sigmoid Function di Lecture's Notes salah (http://mathworld.wolfram.com/SigmoidFunction.html)
        Z(h2,1) = Z_in(h2);                                                                           %11.Kirim nilai aktivasi sebagai input untuk unit output
      end                                                                                         %12.End for

      
      %Hidden layer 1 ke output
      for h3 = 1:l                                                                                   %13.For setiap unit output Ym
        Y_in(h3) = W0(1,h3) + sum(Z.*W(:,h3));                                                        %14.Hitung  semua  sinyal  input  yang  didapatkan  dari  layer  sebelumnya  dengan  bobot dan biasnya sesuai (1.4)
        Y(h3,1) = 1/(1+exp(-Y_in(h3)));                                                               %15.Hitung nilai aktivasi setiap unit output sebagai output dari jaringan, sesuai (1.5)
      end                                                                                         %16.End for
      
      %Mencoba menjawab dengan mengambil nilai tertinggi di suatu output
      value = Y(1,1);
      jawaban = 1;
      for h4 = 2:l
        if Y(h4,1) > value
          value = Y(h4,1);
          jawaban = h4;
        end
      end
      
      %Mencocokkan jawaban
      if jawaban == dataTesting(h,5)+1
        jumlah_sukses_testing(dataTesting(h,5)+1) = jumlah_sukses_testing(dataTesting(h,5)+1) + 1;
      end
    
    end
 
    %Menampilkan data MSSE
    fprintf ('RMSE terrendah yang dapat dihasilkan sebesar %.2f\n', Best_MSSE)
    fprintf ('Epoch dengan MSSE terbaik berada di Epoch %d\n', Best_Epoch)
    
    %Menampilkan data recognition rate dataset Training
    fprintf ('\n')
    fprintf ('\n')
    disp ('Hasil Recognition Dataset Training:')
    for t = 1:l
      jumlah_dataTraining_kelas(t) = jumlah_dataTraining/l;
      recognition_rate_training_class(t) = jumlah_sukses_training(t)*100/jumlah_dataTraining_kelas(t);
      fprintf ('Kelas %d :\n', t-1)
      fprintf ('Jumlah Benar = %d\n', jumlah_sukses_training(t))
      fprintf ('Jumlah Keseluruhan = %d\n', jumlah_dataTraining_kelas(t))
      fprintf ('Recognition Rate = %.2f%%\n', recognition_rate_training_class(t))
      fprintf ('\n')
    end
    
    %Menampilkan data recognition rate dataset Testing
    disp ('Hasil Recognition Dataset Testing:')
    for t = 1:l
      jumlah_dataTesting_kelas(t) = jumlah_dataTesting/l;
      recognition_rate_testing_class(t) = jumlah_sukses_testing(t)*100/jumlah_dataTesting_kelas(t);
      fprintf ('Kelas %d :\n', t-1)
      fprintf ('Jumlah Benar = %d\n', jumlah_sukses_testing(t))
      fprintf ('Jumlah Keseluruhan = %d\n', jumlah_dataTesting_kelas(t))
      fprintf ('Recognition Rate = %.2f%%\n', recognition_rate_testing_class(t))
      fprintf ('\n')
    end



disp('SELESAI')
%Referensi :
%https://www.kaggle.com/c/digit-recognizer/discussion/4667
%http://mathworld.wolfram.com/SigmoidFunction.html
%https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
%https://towardsdatascience.com/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-1-regrression-metrics-3606e25beae0