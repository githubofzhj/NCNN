import tensorflow as tf
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import numpy as np
from tensorflow.contrib import learn
import random
from sklearn import metrics
from cnnmodel import *
from read_data import *
from sklearn.model_selection import KFold
# settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# base = 'tweet'
base = 'ace05'
flags.DEFINE_string('model', 'CNN', 'model')
flags.DEFINE_string('task', 'aspect', 'task') # 'aspect' or 'term'
flags.DEFINE_integer('run', 1, 'first run')
flags.DEFINE_integer('batch_size', 50, 'Batch_size')
flags.DEFINE_integer('epochs', 200, 'epochs')
flags.DEFINE_integer('classes', 34, 'class num')
flags.DEFINE_integer('aspnum', 5, 'asp num')
flags.DEFINE_integer('nhop', 4, 'hops num')
flags.DEFINE_integer('hidden_size', 300, 'number of hidden units')
flags.DEFINE_integer('embedding_size', 300, 'embedding_size')
flags.DEFINE_integer('word_output_size', 300, 'word_output_size')
# flags.DEFINE_string('checkpoint_maxacc', 'data/' + base + '/checkpoint_maxacc/', 'checkpoint dir')
# flags.DEFINE_string('checkpoint_minloss', 'data/' + base + '/checkpoint_minloss/', 'checkpoint dir')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('attRandomBase', 0.01, 'attRandomBase')
flags.DEFINE_float('biRandomBase', 0.01, 'biRandomBase')
flags.DEFINE_float('aspRandomBase', 0.01, 'aspRandomBase')
flags.DEFINE_float('seed', 2018, 'random_seed')
flags.DEFINE_float('max_grad_norm', 5.0, 'max-grad-norm')
flags.DEFINE_float('dropout_keep_proba',0.5, 'dropout_keep_proba')
flags.DEFINE_string('embedding_file', 'data/' + base + '/embedding.npy', 'embedding_file')
# flags.DEFINE_string('testdata', 'data/' + base + '/test', 'testdata')
flags.DEFINE_string('devdata', 'data/' + base + '/dev.txt', 'devdata')
flags.DEFINE_string('aspcat', 'data/' + base + '/aspcat.txt', 'aspcatdata')
flags.DEFINE_string('device', '/cpu:0', 'device')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_string('filter_sizes', '3,4,5', "Comma-separated filter sizes (default: '3,4,5')")

flags.DEFINE_string("train_data", "data/ABSA-15_Laptops_Train_Data.xml", "train gold data set path [./data/Laptop_Train_v2.xml]")
flags.DEFINE_string("test_data", "data/ABSA15_Laptops_Test.xml", "test gold data set path [./data/Laptops_Test_Gold.xml]")
flags.DEFINE_string("embeddingfile", "data/glove.840B.300d.txt", "embeddingfile")
flags.DEFINE_float("embedding_random", 0.25, "embeddingfile")
flags.DEFINE_string("wrongfile", "data/wrongfile", "wrongfile")

checkpoint_maxf1_dir = './dataprocess/data/' + base + '/checkpoint_maxf1/' + FLAGS.model + '/'
checkpoint_minloss_dir = './dataprocess/data/'+ base + '/checkpoint_minloss/' + FLAGS.model + '/'
tflog_dir = os.path.join('./dataprocess/data/' + base + '/', 'tflog')

def batch_iterator(datalist, batch_size, model='train',):
    w=15#window_size=2w+1
    xb = []
    yb = []
    tb = []
    pb = []
    num = len(datalist)//batch_size
    list1 = list(range(num))
    if model == 'train':
        random.shuffle(list1)
    max_length=len(datalist[0][0])#第一个句子补齐后长度
    for i in list1:
        #print i
        for j in range(batch_size):
            index = i * batch_size + j
            x = datalist[index][0]
            y = datalist[index][1]
            # t = datalist[index][2]

            #x左右段补w个0
            x_t =[]
            for m in range(w):
                x_t.append(0)
            x_t.extend(x)
            for m in range(w):
                x_t.append(0)


            for k in range(datalist[index][2]):# 句子长度

                ##让第k个单词作为中间
                x_=x_t[k:2*w+k+1]

                xb.append(x_)
                pb.append([abs(i) for i in range(-w,w+1)])
            yb.extend(y)
            # tb.append(t)
        # if model == 'train':
        yield xb, yb,pb
        xb, yb, pb= [], [], []
    if len(datalist) % batch_size != 0:
        for i in range(num * batch_size, len(datalist)):
            x = datalist[i][0]
            y = datalist[i][1]

            # x左右段补w个0
            x_t = []
            for m in range(w):
                x_t.append(0)
            x_t.extend(x)
            for m in range(w):
                x_t.append(0)

            for k in range(datalist[i][2]):
                ##让第k个单词作为中间
                x_ = x_t[k:2 * w + k + 1]

                xb.append(x_)
                pb.append([abs(i) for i in range(-w, w + 1)])
            yb.extend(y)#y是一维的
        # if model == 'train':
        yield xb, yb,pb
def myprf(pre, lab):
    _pre=[]
    _lab=[]
    for pre1 in pre:
        _pre.extend(pre1)
    for lab1 in lab:
        _lab.extend(lab1)
    ct, p1, r1 = 0, 0, 0
    for i in range(len(_pre)):
        if _pre[i]!=0:
            p1+=1
        if _pre[i]!=0 and _pre[i]==_lab[i]:
            ct+=1
        if _lab[i]!=0 :
            r1+=1
    if ct == 0 or p1 == 0 or r1 == 0:
        return 0.0, 0.0, 0.0
    else:
        r = 1.0 * ct / r1
        p= 1.0 * ct / p1
        f1 = 2.0 * p * r / (p + r)
        return p, r, f1

def xiaoya_prf(pre, lab):
    _pre=[]
    _lab=[]
    for pre1 in pre:
        _pre.extend(pre1)
    for lab1 in lab:
        _lab.extend(lab1)
    ct, p1, r1 = 0, 0, 0
    for i in range(len(_pre)):
        if _lab[i]!=0 :
            r1+=1
            if _pre[i] != 0:
                p1 += 1
            if _pre[i] == _lab[i]:
                ct += 1
    if ct == 0 or p1 == 0 or r1 == 0:
        return 0.0, 0.0, 0.0
    else:
        r = 1.0 * ct / r1
        p= 1.0 * ct / p1
        f1 = 2.0 * p * r / (p + r)
        return p, r, f1


def myprf_sequence(pre,lab):
    ct, p1, r1 = 0, 0, 0
    new_event=False
    for k,sentence_pre in enumerate(pre):
        sentence_lab = lab[k]

        start=0
        pre_list=[]

        for i in range(len(sentence_pre)):
            if(sentence_pre[i]==sentence_pre[start]):
                continue
            if(sentence_pre[start]!=0):
                pre_list.append((start,i-1,sentence_pre[start]))
            start=i
        if (sentence_pre[start] != 0):
            pre_list.append((start, len(sentence_pre) - 1, sentence_pre[start]))

        start = 0
        lab_list=[]
        for i in range(len(sentence_lab)):
            if (sentence_lab[i] == sentence_lab[start]):
                continue
            if (sentence_lab[start] != 0):
                lab_list.append((start, i - 1, sentence_lab[start]))
            start = i
        if (sentence_lab[start] != 0):
            lab_list.append((start, len(sentence_lab) - 1, sentence_lab[start]))

        for _pre in pre_list:
            for _lab in lab_list:
                if(_pre == _lab):
                    ct+=1

        # print(pre_list)
        # print(lab_list)
        p1+=len(pre_list)
        r1+=len(lab_list)



    if ct == 0 or p1 == 0 or r1 == 0:
        return 0.0, 0.0, 0.0
    else:
        r = 1.0 * ct / r1
        p= 1.0 * ct / p1
        f1 = 2.0 * p * r / (p + r)
        return p, r, f1

def evaluate(session, model, datalist, base='none', op='none', source_word2idx={}):
    predictions = []
    labels = []
    categories=[]
    attprob = []
    lossAll = 0.0
    for x, y ,p in batch_iterator(datalist, FLAGS.batch_size, 'val'):
        labels.append(y)
        # pred, loss, att = session.run([model.prediction, model.loss, model.attprob], model.get_feed_data(x, t, y, a, e=None, is_training=False))
        pred, loss = session.run([model.prediction, model.loss], model.get_feed_data(x, y,p,e=None, is_training=False))
        predictions.append(pred)
        # attprob.extend(att)
        lossAll += loss
        # if len(labels) == FLAGS.batch_size:
        #     print labels
        #     print predictions
    #divide_cate_result(categories,labels,predictions)
    n = 0
    wronglist=[]

    ##
    lab_print=[]
    pre_print=[]
    for j,_ in enumerate(labels):
        for i in range(len(labels[j])):
            if( labels[j][i]!=0 or predictions[j][i]!=0):
                lab_print.append(labels[j][i])
                pre_print.append(predictions[j][i])
    lab_s=''
    for i in lab_print:
        lab_s+="%2d"%i+' '
    pre_s=''
    for i in pre_print:
        pre_s+="%2d"%i+' '
    print("lab_print:"+lab_s)
    print("pre_print:"+pre_s)

    # #去除label和pre全部为none的
    # none_type_list=[]
    # for i in range(len(labels)):
    #     if labels[i] == 0 and predictions[i]==0:
    #         none_type_list.append(i)
    # labels_,predictions_=[],[]
    # for i in range(len(labels)):
    #     if i not in none_type_list:
    #         labels_.append(labels[i])
    #         predictions_.append(predictions[i])
    # labels,predictions=labels_,predictions_
    # print("after len(labels):"+str(len(labels)))
    # print("after len(predictions):" + str(len(predictions)))
    # p, r, f, sup = metrics.precision_recall_fscore_support(labels, predictions, average='micro')
    # return float(n)/float(len(labels)), lossAll, p, r, f, predictions, attprob
    p1,r1,f1=xiaoya_prf(predictions, labels)
    p2,r2,f2=myprf(predictions, labels)
    return  lossAll, p1, r1, f1,p2,r2,f2
#cnn需要补齐
def padorcut2maxlength(datalist, padorcut_document_length):
    word_pad=[]
    trigger_sparse_pad=[]
    for i in range(len(datalist)):
        if len(datalist[i][0])<=padorcut_document_length:
            for j in range(len((datalist[i][0])),padorcut_document_length):
                datalist[i][0].append(0)
                #datalist[i][1].append(0)#不能补充o，否则后面batch iter时候长度不一致的
            word_pad.append(datalist[i][0])
            # trigger_sparse_pad.append(datalist[i][1])
        else:
            word_pad.append(datalist[i][0][:padorcut_document_length])
            # trigger_sparse_pad.append(datalist[i][1][:padorcut_document_length])
    return  word_pad,0

def sklearn_Kfold(kf,train_x,train_c,train_y,num):
    i=0
    for train_index, dev_index in kf.split(train_x):
        if(i==num):
            return  np.array(train_x)[train_index], np.array(train_c)[train_index],np.array(train_y)[train_index], np.array(train_x)[dev_index], np.array(train_c)[dev_index],np.array(train_y)[dev_index]
        i+=1
    raise ValueError("Unknown flodnum k less than num!")
def countlen(x):
    x_len=[]
    for line in x:
        x_len.append(len(line))
    return  x_len
def get_dev(train_x,train_c,train_y,dev_nm):
    train_all = list(zip(train_x, train_c, train_y))
    random.shuffle(train_all)  # 先同步随机打乱
    ktrain_all=train_all[0:(len(train_all)-dev_nm)]
    kdev_all=train_all[(len(train_all)-dev_nm):]
    train_x[:], train_c[:], train_y[:] = zip(*ktrain_all)
    dev_x, dev_c, dev_y=list(),list(),list()
    dev_x[:],dev_c[:],dev_y[:]=zip(*kdev_all)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print ("---  new folder...  ---")
        print ("---  OK  ---")

    else:
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            os.remove(c_path)
        print ("---  There is this folder! And clearned！ ---")
def get_dev(train_x,train_c,train_y,dev_nm):
    train_all = list(zip(train_x, train_c, train_y))
    random.shuffle(train_all)  # 先同步随机打乱
    ktrain_all=train_all[0:(len(train_all)-dev_nm)]
    kdev_all=train_all[(len(train_all)-dev_nm):]
    train_x[:], train_c[:], train_y[:] = zip(*ktrain_all)
    dev_x, dev_c, dev_y=list(),list(),list()
    dev_x[:],dev_c[:],dev_y[:]=zip(*kdev_all)
    return train_x, train_c, train_y, dev_x, dev_c, dev_y

def store_preprocessed_file(outdir, data_list):
    print("store data")
    for i in range(len(data_list)):
        with open(outdir + str(i)+".json", 'w') as f:#保存第i个需要保存的数据类型
            json.dump(data_list[i], f, indent=4)

def load_preprocessed_file(outdir, data_list):
    print("load data")
    for i in range(len(data_list)):
        with open(outdir + str(i)+".json", 'r',encoding='utf-8') as f:#保存第i个需要保存的数据类型
            data_list[i] = json.load(f)

def train():
    preprocessed_dir='./dataprocess/data/preprocessed/'

    if( not os.path.exists(preprocessed_dir)):
        print("data preprocess")
        os.makedirs(preprocessed_dir)

        trainfile='./dataprocess/data/Englishsplit/trainevent.json'
        testfile = './dataprocess/data/Englishsplit/testevent.json'
        devfile = './dataprocess/data/Englishsplit/devevent.json'
        ##getword##
        train_source_words, train_target_words, trian_trigger_words_locations,train_max_sent_len,train_source_len=read_event_data(trainfile,'train')
        train_word_num=0
        for  i in train_source_words:
            train_word_num+=len(i)
        print(train_word_num)

        dev_source_words, dev_target_words, dev_trigger_words_locations,dev_max_sent_len,dev_source_len = read_event_data(devfile,'dev')
        test_source_words, test_target_words, test_trigger_words_locations,test_max_sent_len ,test_source_len= read_event_data(testfile,'test')

        print("train"+str(len(train_source_words)))
        print("dev"+str(len(dev_source_words)))
        print("test"+str(len(test_source_words)))

        max_document_length=max([train_max_sent_len,dev_max_sent_len,test_max_sent_len])
        print("max_document_length:"+str(max_document_length))

        ##getwordwid##
        source_words, target_words=train_source_words+dev_source_words+test_source_words,train_target_words+dev_target_words+test_target_words
        source_word2idx, target_word2idx, source_idx2word, target_idx2word=getall_word2id(source_words,target_words)
        print("trigger_word2id:")
        print(target_word2idx)
        ###train test dev data
        train_source_id, train_target_id,trian_trigger_id_sparse=data2id(train_source_words, train_target_words, source_word2idx, target_word2idx,trian_trigger_words_locations)
        dev_source_id, dev_target_id ,dev_trigger_id_sparse= data2id(dev_source_words, dev_target_words, source_word2idx, target_word2idx,dev_trigger_words_locations)
        test_source_id,test_target_id ,test_trigger_id_sparse= data2id(test_source_words, test_target_words, source_word2idx, target_word2idx,test_trigger_words_locations)


        print("dev_source_words:"+str(dev_source_words[0]))
        print("dev_target_words:"+str(dev_target_words[0]))
        print("dev_trigger_words_locations:"+str(dev_trigger_words_locations[0]))
        print("dev_source_len:"+str(dev_source_len[0]))
        print("dev_trigger_id_sparse:")
        print(dev_trigger_id_sparse)

        del train_source_words
        del test_source_words
        del dev_source_words
        ###pad
        # pad_sentence_length = 100
        # train_source_id, trian_trigger_id_sparse__ = padorcut2maxlength(list(zip(train_source_id,trian_trigger_id_sparse)), pad_sentence_length)
        # dev_source_id, dev_trigger_id_sparse__ = padorcut2maxlength(list(zip(dev_source_id,dev_trigger_id_sparse)),
        #                                                               pad_sentence_length)
        # test_source_id, test_trigger_id_sparse__ = padorcut2maxlength(list(zip(test_source_id ,test_trigger_id_sparse)),
        #                                                               pad_sentence_length)
        # max_document_length = pad_sentence_length


        ##getembedding
        print("embedding")
        embeddingfile='./dataprocess/data/Englishsplit/word2vec.json'
        word_embedding=get_word_embeddingMatrix(embeddingfile,source_word2idx, 300, 0.25)

        '''''json dir'''''
        data_list=[train_source_id, train_target_id,trian_trigger_id_sparse,dev_source_id, dev_target_id ,dev_trigger_id_sparse, \
                    test_source_id, test_target_id, test_trigger_id_sparse,train_source_len,test_source_len,dev_source_len,word_embedding
                    ]
        store_preprocessed_file(outdir=preprocessed_dir,data_list=data_list)

    else:



        data_list = [[] for _ in range(13)]


        load_preprocessed_file(preprocessed_dir,data_list)

        train_source_id, train_target_id, trian_trigger_id_sparse, dev_source_id, dev_target_id, \
        dev_trigger_id_sparse, \
        test_source_id, test_target_id, test_trigger_id_sparse, train_source_len, test_source_len, \
        dev_source_len, word_embedding=data_list


    ##tf
    tf.reset_default_graph()
    fres = open('./dataprocess/data/' + 'ace_' + FLAGS.model + '.txt', 'w')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as s:

        # model = MemNN(FLAGS, word_embedding, s)
        cnn_sentence_len=31
        model = CNN(FLAGS, np.array(word_embedding), s, cnn_sentence_len, [2, 3, 4, 5], 150)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # '''
        if FLAGS.run == 1:
            s.run(tf.global_variables_initializer())
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_minloss_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(s, ckpt.model_checkpoint_path)
        # summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())
        cost_val = []
        maxvalacc = 0.0
        maxvalf1=0.0
        minloss = 10000.0
        valnum = 0
        stepnum = 0
        is_earlystopping = False


        istrainflag = 1
        if (istrainflag == 0):
            print('not  trian! just test!')
        if (istrainflag == 1):
            mkdir(checkpoint_minloss_dir)
            mkdir(checkpoint_maxf1_dir)
            for i in range(FLAGS.epochs):
                starttime = datetime.datetime.now()
                print("Epoch:", '%04d' % (i + 1))
                fres.write("Epoch: %02d \n" % (i + 1))
                for j, (x, y,p) in enumerate(
                        batch_iterator(list(zip(train_source_id, trian_trigger_id_sparse,train_source_len)), FLAGS.batch_size, 'train')):
                    fd = model.get_feed_data(x, y,p, e=None)
                    step, labels, prediction, loss, accuracy, _, = s.run([
                        model.global_step,
                        model.labels,
                        model.prediction,
                        model.loss,
                        model.accuracy,
                        model.train_op,
                    ], fd)
                    # print hid
                    stepnum = stepnum + 1
                    # print labels
                    # print prediction
                    # trainloss_epoch += loss
                    # if stepnum % 100 == 0:
                    # print("Step:", '%05d' % stepnum, "train_loss=", "{:.5f}".format(loss),
                    #       "train_loss_senti=", "{:.5f}".format(loss_senti), "train_loss_cla=", "{:.5f}".format(loss_cla),
                    #       "train_acc=", "{:.5f}".format(accuracy),"time=", "{:.5f}".format(time.time() - t0))
                    # if stepnum % (64000/FLAGS.batch_size) == 0:
                    # if stepnum % (64000 / FLAGS.batch_size) == 0:
                # valnum += 1
                dev_ml_loss, dev_mlp, dev_mlr, dev_mlf ,dev_mlp2, dev_mlr2, dev_mlf2= evaluate(s, model, list(zip(dev_source_id, dev_trigger_id_sparse,dev_source_len)))
                print("dev set results loss= %.5f  \n|| sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
                dev_ml_loss,  dev_mlp, dev_mlr, dev_mlf,dev_mlp2, dev_mlr2, dev_mlf2))

                fres.write("dev set results loss= %.5f  \n|| sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
                dev_ml_loss,  dev_mlp, dev_mlr, dev_mlf,dev_mlp2, dev_mlr2, dev_mlf2))


                test_ml_loss, test_mlp, test_mlr, test_mlf,test_mlp2, test_mlr2, test_mlf2 = evaluate(s, model, list(zip(test_source_id,test_trigger_id_sparse,test_source_len)))
                print("Test set results loss= %.5f  \n|| sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
                test_ml_loss, test_mlp, test_mlr, test_mlf,test_mlp2, test_mlr2, test_mlf2))

                fres.write("Test set results loss= %.5f  \n|| sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
                test_ml_loss, test_mlp, test_mlr, test_mlf,test_mlp2, test_mlr2, test_mlf2))

                if dev_mlf > maxvalf1:
                    maxvalf1 = dev_mlf
                    saver.save(s, checkpoint_maxf1_dir + 'model.ckpt', global_step=step)
                if dev_ml_loss < minloss:
                    minloss = dev_ml_loss
                    saver.save(s, checkpoint_minloss_dir + 'model.ckpt', global_step=step)

                endtime = datetime.datetime.now()
                print("time_seconds:%d\n"%(endtime - starttime).seconds)

                # if valnum > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                #     is_earlystopping = True
                #     print("Early stopping...")
                #     fres.write("Early stopping...\n")
                #     break
                # print("Epoch:", '%04d' % (i + 1), " train_loss=", "{:.5f}".format(trainloss_epoch))
                # fres.write("Epoch: %06d train_loss= %.5f \n" % ((i + 1), trainloss_epoch))
                # if is_earlystopping:
                #     break
            print("Optimization Finished!")
            fres.write("Optimization Finished!\n")

        # Testing
        # test_x, test_y, test_t, test_p = utils.load_data(FLAGS.testdata, FLAGS.task)
        # test_acc, test_loss = evaluate(s, model, list(zip(test_x, test_y, test_u, test_p)))
        # print("Test set results based last pamameters: cost= %.5f accuracy= %.5f \n"%(test_loss,test_acc))
        # fres.write("Test set results based last pamameters: cost= %.5f accuracy= %.5f \n"%(test_loss,test_acc))

        ckpt = tf.train.get_checkpoint_state(checkpoint_maxf1_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)
        test_vma_loss, map, mar, maf ,map2, mar2, maf2= evaluate(s, model, list(zip(test_source_id,test_trigger_id_sparse,test_source_len)), 'maxacc',
                                                              'end', source_word2idx)
        print(
            "Test set results based max-val-f1 pama: cost= %.5f  \n||sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
                test_vma_loss, map, mar, maf, map2, mar2, maf2))
        fres.write(
            "Test set results based max-val-f1 pama: cost= %.5f  \n||sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
                test_vma_loss, map, mar, maf, map2, mar2, maf2))
        # fp = open('data/' + base + '/pre_memnn1.txt', 'w')
        # for k in range(len(tpre)):
        #     fp.write(str(tpre[k]) + '\n')
        #
        ckpt = tf.train.get_checkpoint_state(checkpoint_minloss_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)
        # test_ml_acc, test_ml_loss, mlp, mlr, mlf, tpre, attprob = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)))
        test_ml_loss, mlp, mlr, mlf ,mlp2, mlr2, mlf2= evaluate(s, model, list(zip(test_source_id,test_trigger_id_sparse,test_source_len)), 'minloss',
                                                            'end', source_word2idx)
        print(
            "Test set results based min-loss-acc pama: cost= %.5f  \n||sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
            test_ml_loss,  mlp, mlr, mlf ,mlp2, mlr2, mlf2))
        fres.write(
            "Test set results based min-loss-acc pama: cost= %.5f  \n||sequence:precision= %.5f recall= %.5f f-score= %.5f \n||word: precision= %.5f recall= %.5f f-score= %.5f\n" % (
            test_ml_loss,  mlp, mlr, mlf ,mlp2, mlr2, mlf2))


def main():
    train()

if __name__ == '__main__':
    # pre=[[0,1,1,2,3,4,5,0,2,0,0,1,1,0],[1,2,2,0,0,0]]
    # lab=[[0,1,1,2,3,4,5,0,2,0,0,1,2,0],[0,0,2,0,0,0]]
    # print(myprf_sequence(pre, lab))
    main()
