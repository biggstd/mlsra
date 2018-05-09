import tensorflow as tf
import pandas as pd
import glob
import os


'''
Your dataset.
'''

FKPM_GLOB = '/media/tylerbiggs/genomic1/data/**/*.fpkm'

FKPM_FILE_LIST = glob.glob(FKPM_GLOB, recursive=True)

SAMPLE_ANNOTATIONS = '/media/tylerbiggs/genomic1/data/sample_annots.txt'

LABEL_DF = pd.read_table(SAMPLE_ANNOTATIONS)

FPKM_ARRAYS = glob.glob('/media/tylerbiggs/genomic1/data/**/*.fpkm.npy',
                        recursive=True)


def build_fpkm_arrays(fpkm_file):
    '''Builds a set of numpy arrays from a given FPKM file.'''
    new_df = pd.read_table(fpkm_file, sep=r'\s+', header=None)
    sra_id = os.path.basename(fpkm_file).split('_')[0]
    return sra_id, new_df.as_matrix().T[1]


def get_label(fpkm_filepath, label_df=LABEL_DF):
    '''Get a label from the dataframe.'''
    sra_id = os.path.basename(fpkm_filepath).split('_')[0]
    label_index = label_df.index[label_df['Sample'] == sra_id]
    label = label_df['Treatment'].iloc[label_index].values[0]
    return label




xs = [ 0.00,  1.00,  2.00, 3.00, 4.00, 5.00, 6.00, 7.00] # Features
ys = [-0.82, -0.94, -0.12, 0.26, 0.39, 0.64, 1.02, 1.00] # Labels
'''
Initial guesses, which will be refined by TensorFlow.
'''
m_initial = -0.5 # Initial guesses
b_initial =  1.0

'''
Define free variables to be solved.
'''
m = tf.Variable(m_initial) # Parameters
b = tf.Variable(b_initial)

'''
Define placeholders for big data.
'''
_BATCH = 8 # Use only eight points at a time.
xs_placeholder = tf.placeholder(tf.float32, [_BATCH])
ys_placeholder = tf.placeholder(tf.float32, [_BATCH])

'''
Define the error between the data and the model as a tensor (distributed computing).
'''
ys_model = m*xs_placeholder+b # Tensorflow knows this is a vector operation
total_error = tf.reduce_sum((ys_placeholder-ys_model)**2) # Sum up every item in the vector

'''
Once cost function is defined, create gradient descent optimizer.
'''
optimizer_operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_error) # Does one step

'''
Create operator for initialization.
'''
initializer_operation = tf.global_variables_initializer()

'''
All calculations are done in a session.
'''
with tf.Session() as session:

	session.run(initializer_operation) # Call operator

	_EPOCHS = 10000 # Number of "sweeps" across data
	for iteration in range(_EPOCHS):
		random_indices = np.random.randint(len(xs), size=_BATCH) # Randomly sample the data
		feed = {
			xs_placeholder: xs[random_indices],
			ys_placeholder: ys[random_indices]
		}
		session.run(optimizer_operation, feed_dict=feed) # Call operator

	slope, intercept = session.run((m, b)) # Call "m" and "b", which are operators
	print('Slope:', slope, 'Intercept:', intercept)
