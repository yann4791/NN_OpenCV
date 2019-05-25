#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>

static int
read_num_class_data(const char* filename, int var_count,
CvMat** data, CvMat** responses)
{
	const int M = 1024;
	FILE* f = fopen(filename, "rt");
	CvMemStorage* storage;
	CvSeq* seq;
	char buf[M + 2];
	float* el_ptr;
	CvSeqReader reader;
	int i, j;

	if (!f)
		return 0;

	el_ptr = new float[var_count + 1];
	storage = cvCreateMemStorage();
	seq = cvCreateSeq(0, sizeof(*seq), (var_count + 1)*sizeof(float), storage);

	for (;;)
	{
		char* ptr;
		if (!fgets(buf, M, f) || !strchr(buf, ','))
			break;
		el_ptr[0] = buf[0];
		ptr = buf + 2;
		for (i = 1; i <= var_count; i++)
		{
			int n = 0;
			sscanf(ptr, "%f%n", el_ptr + i, &n);
			ptr += n + 1;
		}
		if (i <= var_count)
			break;
		cvSeqPush(seq, el_ptr);
	}
	fclose(f);

	*data = cvCreateMat(seq->total, var_count, CV_32F);
	*responses = cvCreateMat(seq->total, 1, CV_32F);

	cvStartReadSeq(seq, &reader);

	for (i = 0; i < seq->total; i++)
	{
		const float* sdata = (float*)reader.ptr + 1;
		float* ddata = data[0]->data.fl + var_count*i;
		float* dr = responses[0]->data.fl + i;

		for (j = 0; j < var_count; j++)
			ddata[j] = sdata[j];
		*dr = sdata[-1];
		CV_NEXT_SEQ_ELEM(seq->elem_size, reader);
	}

	cvReleaseMemStorage(&storage);
	delete[] el_ptr;
	return 1;
}



static
int build_mlp_classifier(char* data_filename)
{
	const int class_count = 26; // количество классов
	CvMat* data = 0;  // все данные
	CvMat train_data;   // для обучения входные данные
	CvMat* responses = 0;  // все выходные данные
	CvMat* mlp_response = 0;

	int ok = read_num_class_data(data_filename, 16, &data, &responses);  // чтение из файла 16000 сигналов
	int nsamples_all = 0, ntrain_samples = 0; // для тестирования обучения
	int i, j;
	double train_hr = 0, test_hr = 0; // правильная классификация данных обучения и данных тестирования
	CvANN_MLP mlp;  // сеть

	if (!ok)
	{
		printf("Could not read the database %s\n", data_filename);
		return -1;
	}

	printf("The database %s is loaded.\n", data_filename);
	nsamples_all = data->rows;
	ntrain_samples = (int)(nsamples_all*0.8);  // 80% от загруженных

	// Create  MLP classifier
	
	CvMat* new_responses = cvCreateMat(ntrain_samples, class_count, CV_32F); // известные выходы для обучения

		
		printf("Unrolling the responses...\n");
		for (i = 0; i < ntrain_samples; i++)
		{
			int cls_label = cvRound(responses->data.fl[i]) - 'A';  // номер буквы
			float* bit_vec = (float*)(new_responses->data.ptr + i*new_responses->step);
			for (j = 0; j < class_count; j++)
				bit_vec[j] = 0.f;
			bit_vec[cls_label] = 1.f;  // 1 только у своего класса
		}
		cvGetRows(data, &train_data, 0, ntrain_samples);

		// 2. train classifier
		int layer_sz[] = { data->cols, 100, 100, class_count }; // количество нейронов в обычном массиве
		CvMat layer_sizes =
			cvMat(1, (int)(sizeof(layer_sz) / sizeof(layer_sz[0])), CV_32S, layer_sz); // количество нейронов
		mlp.create(&layer_sizes);  // создание сети
		printf("Training the classifier (may take a few minutes)...\n");
// методы обучения
		/*
		int method = CvANN_MLP_TrainParams::BACKPROP;
		double method_param = 0.001;
		int max_iter = 300;*/
		int method = CvANN_MLP_TrainParams::RPROP;
		double method_param = 0.1;
		int max_iter = 1000;

// обучение
		int dd=mlp.train(&train_data, new_responses, 0, 0,
			CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER, max_iter, 0.01),
			method, method_param));
		cvReleaseMat(&new_responses);
		printf(" iter=%d\n",dd);
	

	mlp_response = cvCreateMat(1, class_count, CV_32F);

	// вычисление ошибки на всей выборке
	for (i = 0; i < nsamples_all; i++)
	{
		int best_class;
		CvMat sample;
		cvGetRow(data, &sample, i);
		CvPoint max_loc = { 0, 0 };
		mlp.predict(&sample, mlp_response);
		cvMinMaxLoc(mlp_response, 0, 0, 0, &max_loc, 0);
		best_class = max_loc.x + 'A';

		int r = fabs((double)best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;

		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= (double)(nsamples_all - ntrain_samples);
	train_hr /= (double)ntrain_samples;
	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	// освобождение 
	cvReleaseMat(&mlp_response);
	cvReleaseMat(&data);
	cvReleaseMat(&responses);

	return 0;
}

int main()
{
	
   build_mlp_classifier("./letter-recognition.data");

	system("pause");
	return 0;
}