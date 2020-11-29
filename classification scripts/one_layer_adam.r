# model with:
# 1 hidden layer
# adam optimizer

library(keras)

set.seed(2020)

# creating flags for tuning
FLAGS <- flags(
  flag_integer('neurons1', 25),
  flag_integer('neurons2', 10),
  flag_numeric('dropout1', 0.5),
  #flag_numeric('dropout2', 0.2),
  flag_numeric('lr', 0.01)
)

# Defining the model
build_model <- function(){
  class_mod <- keras_model_sequential()
  class_mod %>%
    layer_dense(units = FLAGS$neurons1, activation = 'relu', input_shape = c(21)) %>% # input layer
    layer_dense(units = FLAGS$neurons2, activation = 'relu') %>%
    layer_dropout(rate = FLAGS$dropout1) %>%
    layer_dense(units = 5, activation = 'softmax') # output layer
  
  # specifying loss function, optimizer and the metrix to display
  class_mod %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(lr = FLAGS$lr),
    metrics = c('accuracy') # metric to output to the user
  )
  
  class_mod
}

class_mod <- build_model()
# halt training when there is no more improvement.
#early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# fit model and store training stats
history <- class_mod %>% fit(
  class_train, target_train,
  epochs = 150, batch_size = 100,
  validation_split = 0.3, shuffle = TRUE#, callbacks = list(early_stop)
)

plot(history)

# evaluate performance on the test data
score <- class_mod %>% evaluate(class_test, target_test, verbose = 0)
save_model_hdf5(class_mod, 'model.h5')

cat('Test loss:', score[1], '\n')
cat('Test accuracy:', score[2], '\n')

