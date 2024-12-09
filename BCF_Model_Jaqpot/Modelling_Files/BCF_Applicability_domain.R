library(ggplot2)
#' A function to compute the inverse Fisher information matrix
#'
#' This function computes the inverse Fisher information matrix given a design matrix
#' @param X is the descriptor matrix with samples on rows and features on columns.
#' @return inverse of Fisher information matrix
#' @keywords fisher information matrix, inverse 
#' @export
#' @examples
#'

inv_fisher = function(X){
  X <- as.matrix(X)
  xtx = t(X) %*% X
  # Add small values to the diagonal so that the inversion can be numerically estimated
  diag(xtx) = diag(xtx) + runif(n = length(diag(xtx)),min = 1e-7,max =  2e-7)
  ixtx = solve(xtx)
  return(ixtx)
}


#' A function to compute the hat matrix
#'
#' This function computes the hat matrix of regression, given the inverse fisher information matrix of the design matrix
#' @param X is the query matrix with samples on rows and features on columns.
#' @param I is the the inverse fisher information matrix 
#' @return hat matrix whose diagonal represents the leverage of the regression
#' @keywords hat matrix, leverages
#' @export
#' @examples
#'

hat_matrix = function(X,I){
  X <-as.matrix(X)
  return(X %*% (I %*% t(X)))
}

#' A function to compute the leverage of a query instance
#'
#' This function computes the leverage value query chemical 
#' @param X is the descriptor matrix with samples on rows and features on columns.
#' @param query_vec is the descriptor row vector of the query instance
#' @return leverage value of a query instance
#' @keywords leverage
#' @export
#' @examples
#'

leverage_val = function(X, query_vec){
  xtx = t(X) %*% X
  # diag(xtx) = diag(xtx) + runif(n = length(diag(xtx)),min = 0.001,max = 0.002)
  ixtx = inv(xtx)
  return(t(query_vec)%*% (ixtx %*% query_vec))
}


#' A function to compute applicability
#'
#' This function computes whether a chemical is within the applicability domain of a model 
#' @param X is the descriptor matrix with samples on rows and features on columns.
#' @param query_vec is the descriptor row vector of the query instance
#' @return leverage value of a query instance
#' @keywords leverage
#' @export
#' @examples
#'

applicability = function(X, query_vec){
  leverage <- leverage_val(X, query_vec)
  p = dim(X)[2]+1
  n = nrow(X)
  hstar = (3*p)/n
  if(leverage>hstar){
    print("Prediction falls outside the applicability domain")
  }else{
    print("Leverage value lower than the warning leverage")
  }
}


#' A function to create the williams plot and compute the percentage of applicability domain
#'
#' This function create the williams plot for a specific model and compute the percentage of training and test point falling in the applicability domain
#' @param X_train is the training dataset matrix with samples on rows and features on columns.
#' @param X_test is the test dataset matrix with samples on rows and features on columns.
#' @param y_train is the vector of the training response.
#' @param y_test is the vector of the test response.
#' @param model is the model for which we want to evaluate the applicability domain
#' @param feats is a vector containing the names of the features
#' @return ADVal is a vector with the percentage of training and test sample falling in the applicability domain
#' @return DTP is a list containing the data to plot the williams plot. In particular it contain the leverages (lev)
#' the residual (res), the colors of training/test points (col) and the h* value (h_star)
#' @keywords williams plot, applicability domain
#' @export
#' @examples
#'

williams_plot = function(x_train, x_test, y_train, y_test, y_pred_train,
                         y_pred_test){
  #  H_train= hat_matrix(rbind(X_train[,feats],X_test[,feats]))
  #H_train= hat_matrix (X_train[,names(beta)])
  Train <- x_train[,-1]
  Test <- x_test[,-1]
  
  I = inv_fisher(Train)
  
  residual_train= (y_train - y_pred_train)
  residual_test= (y_test - y_pred_test)
  s_residual_train = ((residual_train) - mean(residual_train)) / sd(residual_train)
  s_residual_test = (residual_test - mean(residual_test))/ sd(residual_test)
  
  hat_matrix_training <- hat_matrix(Train,I )
  hat_matrix_test <- hat_matrix(Test,I )
  
  leverage_train = diag(hat_matrix_training)
  leverage_test =diag(hat_matrix_test)
  
  feats <- dim(x_train)[2] - 1
  p = feats+1 #features
  n = nrow(Train)#training compounds
  h_star = (3*p)/n
  
  #AD_train = 100 * (sum(leverage_train < h_star & abs(s_residual_train)<3) / length(leverage_train))
  AD_train = 100 * (sum(leverage_train < h_star) / length(leverage_train))
  #AD_test = 100 * (sum(leverage_test < h_star & abs(s_residual_test)<3) / length(leverage_test))
  AD_test = 100 * (sum(leverage_test < h_star ) / length(leverage_test))
  lev=  c(leverage_train,leverage_test)
  res = c(s_residual_train,s_residual_test)
  col = c(rep('Train',nrow(Train)),rep("Test",nrow(Test)))
  
  indexes <- c(x_train$Index, x_test$Index)
  data_to_plot = list(lev=lev,res=res,col=col, indexes = indexes, h_star=h_star)
  
  
  return(list(ADVal = c(AD_train,AD_test), DTP = data_to_plot))
}




#' Plot williams Plot
#'
#' This function perform the random split analysis method
#' @param data_to_plot is a list containing williamps plot data created y the function evaluating_model
#' @keywords plot williams plot
#' @export
#' @examples
#'
plot_wp = function(data_to_plot){
  
  outliers <- data_to_plot[data_to_plot$lev > data_to_plot$h_star | abs(data_to_plot$res) > 3,]
  labels = data_to_plot$indexes[which(data_to_plot$lev > data_to_plot$h_star | abs(data_to_plot$res) > 3)]
  # Create the scatter plot
  p <- ggplot(data_to_plot, aes(x = leverage, y = residuals, color = colour)) +
    
    geom_point(shape = 19, size = 3,  alpha=0.7) +
    ylim(c(min(-4.5, 1.1*data_to_plot$residuals), max(3.5, 1.1*data_to_plot$residuals))) +
    xlim(c(0, max(max(data_to_plot$leverage), data_to_plot$h_star) + 0.2*max(max(data_to_plot$leverage), 
                                                                             data_to_plot$h_star))) +
    labs(x = "Leverage", y = "Standardized Residuals") +
    geom_vline(xintercept = data_to_plot$h_star) +
    geom_hline(yintercept = c(-3, 3), linetype = "dashed") +
    geom_text(data =outliers,
              aes(label = labels),
              nudge_x = 0.2*mean(data_to_plot$leverage), nudge_y = 0*mean(data_to_plot$residuals),
              size = 3, color = "black") +
    scale_y_continuous(limits = c(-4, 4), breaks = c(-3, 0, 3), labels = c(-3, 0, 3))+
    guides(fill = guide_legend(title = 'Data')) +
    scale_color_manual(values = c("Test" = "red", "Train" = "black"))+
    theme_bw() +
    theme(axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          legend.text = element_text(size=16),
          legend.title = element_text(size=16),
          axis.title =element_text(size=16),
          axis.text = element_text(size=16),
          plot.title = element_text(hjust = 0.5, size = 20, face = "bold"))+
    ggtitle("BCF Williams Plot")
  # Print the plot
  print(p)
}

# Load x_train, x_test, y_train and y_test data

x_train <- read.csv('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/x_train.csv')
x_train[x_train == "False"] = as.numeric(0) 
x_train[x_train == "True"] = as.numeric(1)
x_train <- data.frame(lapply(x_train, as.numeric))
names(x_train)[1] <- "Index"


x_test <- read.csv('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/x_test.csv')
x_test[x_test == "False"] = 0
x_test[x_test == "True"] = 1
x_test <- data.frame(lapply(x_test, as.numeric))
names(x_test)[1] <- "Index"


y_train <- read.csv('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/y_train.csv')
y_train <- unlist(as.list(y_train[,2]), use.names = FALSE)[]
y_test <- read.csv('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/y_test.csv')
y_test <- unlist(as.list(y_test[,2]), use.names = FALSE)[]
y_pred_train <- read.csv('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/y_train_pred.csv', header = FALSE)
y_pred_train <- unlist(as.list(y_pred_train[,2]), use.names = FALSE)[]
y_pred_test <- read.csv('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/y_test_pred.csv', header = FALSE)
y_pred_test <- unlist(as.list(y_pred_test[,2]), use.names = FALSE)[]

print(sapply(t(x_train), class))

williams_data <- williams_plot(x_train, x_test, y_train, y_test, y_pred_train, y_pred_test)

data_to_plot <- data.frame(leverage=williams_data$DTP$lev,
                           residuals=williams_data$DTP$res,
                           colour=williams_data$DTP$col,
                           indexes = williams_data$DTP$indexes,
                           h_star=williams_data$DTP$h_star)
pt = plot_wp(data_to_plot)

ggsave("/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/bcf_williams_plot.png", plot = pt, width = 8, height = 6, dpi = 300)


