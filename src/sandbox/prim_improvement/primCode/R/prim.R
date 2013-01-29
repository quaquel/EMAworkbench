
###############################################################################
#### PRIM (Patient rule induction method) for bump-hunting
###############################################################################

###############################################################################
## PRIM (Patient rule induction method)
##
## Parameters
## x - matrix of explanatory variables
## y - vector of response variable
## box.init - initial box (should cover range of x)
## mass.min - min. size of box mass
## y.mean.min - min. threshold of mean of y within a box
## pasting - TRUE - include pasting step (after peeling)
##         - FALSE - don't include pasting
##
## Returns
## list with k fields, one for each box
## each field is in turn is a list with fields
## x - data inside box
## y - corr. response values
## y.mean - mean of y
## box - limits of box
## box.mass - box mass
## num.boxes - total number of boxes with box mean >= y.mean.min
###############################################################################


prim.box <- function(x, y, box.init=NULL, peel.alpha=0.05, paste.alpha=0.01,
                 mass.min=0.05, threshold, pasting=TRUE, verbose=FALSE,
                 threshold.type=0)
{
  if (threshold.type==1 | threshold.type==-1)
  {
    if (missing(threshold))  #missing is a  check whether the argument is provided
      threshold <- mean(y)   #mean returns the mean, appears to be mean across all dims
      prim.temp <- prim.one(x=x, y=threshold.type*y, box.init=box.init,
                         peel.alpha=peel.alpha,
                         paste.alpha=paste.alpha, mass.min=mass.min,
                         threshold.type=threshold.type, 
                         threshold=threshold[1], pasting=pasting,
                         verbose=verbose)
  }
  else
  {
    if (missing(threshold))
      threshold <- c(mean(y), mean(y))
    else if (!missing(threshold))
      if (length(threshold)==1)
        stop("Need both upper and lower values for threshold") #stop code and raise the specified error message
      
    prim.pos <- prim.one(x=x, y=y, box.init=box.init, peel.alpha=peel.alpha,
                         paste.alpha=paste.alpha, mass.min=mass.min,
                         threshold.type=1, threshold=threshold[1],
                         pasting=pasting, verbose=verbose)
    prim.neg <- prim.one(x=x, y=-y, box.init=box.init, peel.alpha=peel.alpha,
                         paste.alpha=paste.alpha,mass.min=mass.min,
                         threshold.type=-1, threshold=threshold[2],
                         pasting=pasting, verbose=verbose)
    prim.temp <- prim.combine(prim.pos, prim.neg)
    
  }
  
  ## re-do prim to ensure that no data points are missed from the `dump' box 
  prim.reg <- prim.temp
  prim.labels <- prim.which.box(x=x, box.seq=prim.reg)
  for (k in 1:prim.reg$num.class)
  {
    primk.ind <- which(prim.labels==k) #indices where prim.labels == K
    prim.reg$x[[k]] <- x[primk.ind,]
    prim.reg$y[[k]] <- y[primk.ind]
    prim.reg$y.mean[k] <- mean(prim.reg$y[[k]])
    prim.reg$mass[k] <- length(prim.reg$y[[k]])/nrow(x) #number of rows
  }
  
  return(prim.reg)
}


prim.one <- function(x, y, box.init=NULL, peel.alpha=0.05, paste.alpha=0.01,
                     mass.min=0.05, threshold, pasting=FALSE, threshold.type=1,
                     verbose=FALSE)
{
  d <- ncol(x) #number of columns
  n <- nrow(x) #number of rows
  k.max <- ceiling(1/mass.min) #ceiling, afronden naar boven: ceiling takes a single numeric argument x and returns a numeric vector containing the smallest integers not less than the corresponding elements of x. 
  num.boxes <- k.max 
 
  ##if (is.vector(x)) x <- as.matrix(t(x))
  y.mean <- mean(y)
  mass.init <- length(y)/n 
  
  if (is.null(box.init))
  {
    box.init <- apply(x, 2, range) #apply range to x, over the columns. Range returns a vector containing the minimum and maximum of all the given arguments. 
    box.diff <- box.init[2,] - box.init[1,]
    box.init[1,] <- box.init[1,] - 10*paste.alpha*box.diff
    box.init[2,] <- box.init[2,] + 10*paste.alpha*box.diff
  }
  
  ## find first box
  k <- 1
  #print(min(y)-0.1*abs(min(y)))
  
  boxk <- find.box(x=x, y=y, box=box.init, peel.alpha=peel.alpha,
                   paste.alpha=paste.alpha, mass.min=mass.min,
                   threshold=min(y)-0.1*abs(min(y)), d=d, n=n, pasting=pasting, verbose=verbose)
		   ##threshold=mean(y) find.box is a function defined in this .r file, further below

  if (is.null(boxk))
  {
    if (verbose)
      warning(paste("Unable to find box", k, "\n")) #concetatenate strings using the specified seperated

    x.prim <- list(x=list(x), y=list(threshold.type*y), y.mean=threshold.type*y.mean, box=list(box.init), box.mass=mass.init, num.class=1, num.hdr.class=1, threshold=mean(y))
    class(x.prim) <- "prim"
    
    return(x.prim)
  }
  else
  {
    if (verbose)
      cat(paste("Found box ", k, ": y.mean=", signif(threshold.type*boxk$y.mean,4), ", mass=", signif(boxk$mass,4), "\n\n", sep=""))
    boxes <- list(x=list(boxk$x), y=list(boxk$y), y.mean=list(boxk$y.mean),
                  box=list(boxk$box), mass=list(boxk$mass))       
				  # cat concetanate and print
				  # signif round to specified digits
				  # list generate an R-style list
	
  }
    
  ## find subsequent boxes
  if (num.boxes > 1)
  {
    boxk <- list(x=boxes$x[[k]], y=boxes$y[[k]], y.mean=boxes$y.mean[[k]],
                 box=boxes$box[[k]], mass=boxes$mass[[k]])

    ## data still under consideration
    x.out.ind.mat <-  matrix(TRUE, nrow=nrow(x), ncol=ncol(x))
	
    for (j in 1:d)
      x.out.ind.mat[,j] <- (x[,j] < boxk$box[1,j]) | (x[,j] > boxk$box[2,j])
	
	x.out.ind <- apply(x.out.ind.mat, 1, sum)!=0
	
	
	
    x.out <- x[x.out.ind,]

    if (is.vector(x.out)) x.out <- as.matrix(t(x.out)) 
    y.out <- y[x.out.ind]
     
	#print(x.out)
	#print(y.out)
    ##box.out <- apply(x.out, 2, range)
    while ((length(y.out)>0) & (k < num.boxes) & (!is.null(boxk))) 
    {
      k <- k+1
	  #print(min(y)-0.1*abs(min(y)))
	  
      boxk <- find.box(x=x.out, y=y.out, box=box.init,
                       peel.alpha=peel.alpha, paste.alpha=paste.alpha,
                       mass.min=mass.min, threshold=min(y)-0.1*abs(min(y)), d=d, n=n,
                       pasting=pasting, verbose=verbose)

      if (is.null(boxk))
      {
        if (verbose)
          cat(paste("Bump", k, "includes all remaining data\n\n"))

        boxes$x[[k]] <- x.out
        boxes$y[[k]] <- y.out
        boxes$y.mean[[k]] <- mean(y.out)
        boxes$box[[k]] <- box.init
        boxes$mass[[k]] <- length(y.out)/n
      }
      else 
      {
        ## update x and y
		if (verbose)
		  cat(paste("Found box ", k, ": y.mean=", signif(threshold.type*boxk$y.mean,4),
                    ", mass=", signif(boxk$mass,4), "\n\n", sep=""))
        
        x.out.ind.mat <- matrix(TRUE, nrow=nrow(x), ncol=ncol(x))
        for (j in 1:d)
          x.out.ind.mat[,j] <- (x[,j] < boxk$box[1,j]) | (x[,j] > boxk$box[2,j])
        
        x.out.ind <- x.out.ind & (apply(x.out.ind.mat, 1, sum)!=0) #sum is sum of vector elements
        x.out <- x[x.out.ind,]
        if (is.vector(x.out)) x.out <- as.matrix(t(x.out))
        y.out <- y[x.out.ind]
     
        boxes$x[[k]] <- boxk$x
        boxes$y[[k]] <- boxk$y
        boxes$y.mean[[k]] <- boxk$y.mean
        boxes$box[[k]] <- boxk$box
        boxes$mass[[k]] <-boxk$mass 
      }
    }   
  }

  ## adjust for negative hdr  
  for (k in 1:length(boxes$y.mean))
  {
    boxes$y[[k]] <- threshold.type*boxes$y[[k]]
    boxes$y.mean[[k]] <- threshold.type*boxes$y.mean[[k]]
  }
  
  ## highest density region

  prim.res <- prim.hdr(prim=boxes, threshold=threshold, threshold.type=threshold.type)
  
  return(prim.res)
         
}

###############################################################################
## Highest density region for PRIM boxes
###############################################################################

prim.hdr <- function(prim, threshold, threshold.type)
{  
  n <- 0
  for (i in 1:length(prim$box))
    n <- n + length(prim$y[[i]])
  
  hdr.ind <- which(unlist(prim$y.mean)*threshold.type >= threshold*threshold.type)
	  
  if (length(hdr.ind) > 0)
    hdr.ind <- max(hdr.ind)
  else
  {
    if (threshold.type==1)
      warning(paste("No prim box found with mean >=", threshold))
    else if (threshold.type==-1)
      warning(paste("No prim box found with mean <=", threshold))
    return()
  }

  
  ## highest density region  
  x.prim.hdr <- list()
  
  for (k in 1:hdr.ind)
  {
    x.prim.hdr$x[[k]] <- prim$x[[k]]
    x.prim.hdr$y[[k]] <- prim$y[[k]]
    x.prim.hdr$y.mean[[k]] <- prim$y.mean[[k]]
    x.prim.hdr$box[[k]] <- prim$box[[k]]
    x.prim.hdr$mass[[k]] <-prim$mass[[k]] 
  }
  
  ## combine non-hdr into a `dump' box
  if (hdr.ind < length(prim$x))
  {
    #cat("making a dumpbox\n")
	
	x.temp <- numeric()
    y.temp <- numeric()
	
	#lump the other boxes together
    for (k in (hdr.ind+1):length(prim$x))
    {
      x.temp <- rbind(x.temp, prim$x[[k]])
      y.temp <- c(y.temp, prim$y[[k]])
    }
    
    x.prim.hdr$x[[hdr.ind+1]] <- x.temp
    x.prim.hdr$y[[hdr.ind+1]] <- y.temp
    x.prim.hdr$y.mean[[hdr.ind+1]] <- mean(y.temp)
    x.prim.hdr$box[[hdr.ind+1]] <- prim$box[[length(prim$x)]]
    x.prim.hdr$mass[[hdr.ind+1]] <- length(y.temp)/n  
  }
  
  x.prim.hdr$num.class <- length(x.prim.hdr$x)
  x.prim.hdr$num.hdr.class <- hdr.ind
  x.prim.hdr$threshold <- threshold
  
  x.prim.hdr$ind <- rep(threshold.type, x.prim.hdr$num.hdr.class)
  
  class(x.prim.hdr) <- "prim"
  
  return(x.prim.hdr)
    
}
 
                        
###############################################################################
## Combine (disjoint) PRIM box sequences - useful for joining
## positive and negative estimates
##
## Parameters
## prim1 - 1st PRIM box sequence
## prim2 - 2nd PRIM box sequence
##
## Returns
## same as for prim()
###############################################################################

prim.combine <- function(prim1, prim2)
{
  M1 <- prim1$num.hdr.class
  M2 <- prim2$num.hdr.class

  if (is.null(M1) & !is.null(M2))
    return (prim2)
  if (!is.null(M1) & is.null(M2))
    return(prim1)
  if (is.null(M1) & is.null(M2))
    return(NULL)

  overlap <- overlap.box.seq(prim1, prim2, rel.tol=0.01)

  x <- numeric()
  y <- vector()
  for (i in 1:prim1$num.class)
  {
    x <- rbind(x, prim1$x[[i]])
    y <- c(y, prim1$y[[i]])
  }  
  

  if (any(overlap[1:M1,1:M2]))
  {
    warning("Class boundaries overlap - will return NULL")
    return(NULL)
  }
  else
  {
    prim.temp <- list()
    for (i in 1:M1)
    {
      prim.temp$x[[i]] <- prim1$x[[i]]
      prim.temp$y[[i]] <- prim1$y[[i]]
      prim.temp$y.mean[[i]] <- prim1$y.mean[[i]]
      prim.temp$box[[i]] <- prim1$box[[i]]
      prim.temp$mass[[i]] <- prim1$mass[[i]]
      prim.temp$ind[[i]] <- 1
    }
    for (i in 1:M2)
    {
      prim.temp$x[[i+M1]] <- prim2$x[[i]]
      prim.temp$y[[i+M1]] <- prim2$y[[i]]
      prim.temp$y.mean[[i+M1]] <- prim2$y.mean[[i]]
      prim.temp$box[[i+M1]] <- prim2$box[[i]]
      prim.temp$mass[[i+M1]] <- prim2$mass[[i]]
      prim.temp$ind[[i+M1]] <- -1
    }
    
    dumpx.ind <- prim.which.box(x, prim1)==prim1$num.class & prim.which.box(x, prim2)==prim2$num.class
    
    prim.temp$x[[M1+M2+1]] <- x[dumpx.ind,]
    prim.temp$y[[M1+M2+1]] <- y[dumpx.ind]
    prim.temp$y.mean[[M1+M2+1]] <- mean(y[dumpx.ind])
    prim.temp$box[[M1+M2+1]] <- prim1$box[[prim1$num.class]]
    prim.temp$mass[[M1+M2+1]] <- length(y[dumpx.ind])/length(y)
    prim.temp$num.class <- M1+M2+1
    prim.temp$num.hdr.class <- M1+M2
    prim.temp$threshold <- c(prim1$threshold, prim2$threshold) 

    class(prim.temp) <- "prim"
  }

  return (prim.temp)
}
   
###############################################################################
## Finds box
##
## Parameters
## x - matrix of explanatory variables
## y - vector of response variable
## box.init - initial box (should cover range of x)
## mass.min - min box mass
## threshold - min box mean
## pasting - TRUE - include pasting step (after peeling)
##         - FALSE - dont include pasting
##
## Returns
## List with fields
## x - data still inside box after peeling
## y - corresponding response values
## y.mean - mean of y
## box - box limits
## mass - box mass
###############################################################################

find.box <- function(x, y, box, peel.alpha, paste.alpha, mass.min, threshold,
                     d, n, pasting, verbose) 
{
  y.mean <- mean(y)
  mass <- length(y)/n
  
  if ((y.mean >= threshold) & (mass >= mass.min))
  {
	boxk.peel <- peel.one(x=x, y=y, box=box, peel.alpha=peel.alpha,
                          mass.min=mass.min, threshold=threshold, d=d, n=n)
  }
  else
    boxk.peel <- NULL

  boxk.temp <- NULL
  #cat("mass: ", boxk.peel$mass, "\n", sep="") 
  #print(boxk.peel$box)
  #cat("mean: ", boxk.peel$y.mean, "\n",sep="")
  
  while (!is.null(boxk.peel))
  { 

	
	boxk.temp <- boxk.peel
	
    boxk.peel <- peel.one(x=boxk.temp$x, y=boxk.temp$y, box=boxk.temp$box,
                          peel.alpha=peel.alpha,
                          mass.min=mass.min, threshold=threshold, d=d, n=n)
  }
  
  if (verbose)
    cat("Peeling completed \n")
  
  if (pasting)
  {
    boxk.paste <- boxk.temp
    
    while (!is.null(boxk.paste))
    {
      boxk.temp <- boxk.paste
      boxk.paste <- paste.one(x=boxk.temp$x, y=boxk.temp$y, box=boxk.temp$box,
                              x.init=x, y.init=y, paste.alpha=paste.alpha,
                              mass.min=mass.min, threshold=threshold, d=d, n=n)      
    }
    if (verbose)
      cat("Pasting completed\n")  
  }
   
  boxk <- boxk.temp

  return(boxk)
}



############################################################################### 
## Peeling stage of PRIM
##
## Parameters
## x - data matrix
## y - vector of response variables
## peel.alpha - peeling quantile
## paste.alpha - peeling proportion
## mass.min - minimum box mass
## threshold - minimum y mean
## d - dimension of data
## n - number of data
## 
## Returns
## List with fields
## x - data still inside box after peeling
## y - corresponding response values
## y.mean - mean of y
## box - box limits
## mass - box mass
###############################################################################

peel.one <- function(x, y, box, peel.alpha, mass.min, threshold, d, n, type=8)
{
  box.new <- box
  mass <- length(y)/n

  if (is.vector(x)) return(NULL)
  
  y.mean <- mean(y)
  y.mean.peel <- matrix(0, nrow=2, ncol=d)
  box.vol.peel <- matrix(0, nrow=2, ncol=d) 
                     
  for (j in 1:d)
  {
    box.min.new <- quantile(x[,j], peel.alpha, type=type)
    box.max.new <- quantile(x[,j], 1-peel.alpha, type=type)
	#cat(box.min.new, box.max.new, "\n", sep="\t")
		
    y.mean.peel[1,j] <- mean(y[x[,j] >= box.min.new])
    y.mean.peel[2,j] <- mean(y[x[,j] <= box.max.new])

    box.temp1 <- box
    box.temp2 <- box
    box.temp1[1,j] <- box.min.new
    box.temp2[2,j] <- box.max.new
    box.vol.peel[1,j] <- vol.box(box.temp1)
    box.vol.peel[2,j] <- vol.box(box.temp2)    
  }
  #print(box.temp1)
  #print(box.temp2)
  #print(box.vol.peel)
    
  y.mean.peel.max.ind <- which(y.mean.peel==max(y.mean.peel, na.rm=TRUE), arr.ind=TRUE)
  ## break ties by choosing box with largest volume

  nrr <- nrow(y.mean.peel.max.ind) 
  if (nrr > 1)
  {
	#cat("more then one box in peel\n")

    box.vol.peel2 <- rep(0, nrr)
    for (j in 1:nrr)
      box.vol.peel2[j] <- box.vol.peel[y.mean.peel.max.ind[j,1],
                                       y.mean.peel.max.ind[j,2]]
	row.ind <- which(max(box.vol.peel2)==box.vol.peel2)

  }
  else
    row.ind <- 1

  y.mean.peel.max.ind <- y.mean.peel.max.ind[row.ind,]
  
  ## peel along dimension j.max
  j.max <- y.mean.peel.max.ind[2]
  
  ## peel lower 
  if (y.mean.peel.max.ind[1]==1)
  {
	box.new[1,j.max] <- quantile(x[,j.max], peel.alpha, type=type)
	print(all.equal(x[,j.max], box.new[1,j.max]))
	x.index <- x[,j.max] >= box.new[1,j.max] 
  }
  ## peel upper 
  else if (y.mean.peel.max.ind[1]==2)
  {
 	box.new[2,j.max] <- quantile(x[,j.max], 1-peel.alpha, type=type)
    x.index <- x[,j.max] <= box.new[2,j.max] 
  }
 
  x.new <- x[x.index,]
  #cat(nrow(x.new), ncol(x.new), "\n", sep="\t")
  
  y.new <- y[x.index]
  mass.new <- length(y.new)/n
  y.mean.new <- mean(y.new)

  ## if min. y mean and min. mass conditions are still true, update
  ## o/w return NULL 

  if ((y.mean.new >= threshold) & (mass.new >= mass.min) & (mass.new < mass))
    return(list(x=x.new, y=y.new, y.mean=y.mean.new, box=box.new, mass=mass.new))
}


###############################################################################
## Pasting stage for PRIM
##
## Parameters
## x - data matrix
## y - vector of response variables
## x.init - initial data matrix (superset of x) 
## y.init - initial response vector (superset of y) 
## peel.alpha - peeling quantile
## paste.alpha - peeling proportion
## mass.min - minimum box mass
## threshold - minimum y mean
## d - dimension of data
## n - number of data
## 
## Returns
##
## List with fields
## x - data still inside box after peeling
## y - corresponding response values
## y.mean - mean of y
## box - box limits
## box.mass - box mass
###############################################################################

paste.one <- function(x, y, x.init, y.init, box, paste.alpha,
                      mass.min, threshold, d, n)
{
  box.new <- box
  mass <- length(y)/n
  y.mean <- mean(y)

  box.init <- apply(x.init, 2, range)
  
  if (is.vector(x)) x <- as.matrix(t(x))
  
  y.mean.paste <- matrix(0, nrow=2, ncol=d)
  mass.paste <- matrix(0, nrow=2, ncol=d)
  box.paste <- matrix(0, nrow=2, ncol=d)
  x.paste1.list <- list()
  x.paste2.list <- list()
  y.paste1.list <- list()
  y.paste2.list <- list()

  box.paste1 <- box
  box.paste2 <- box
  
 
  for (j in 1:d)
  {    
    ## candidates for pasting
    box.diff <- (box.init[2,] - box.init[1,])[j]
	
    box.paste1[1,j] <- box[1,j] - box.diff*paste.alpha
    box.paste2[2,j] <- box[2,j] + box.diff*paste.alpha
    
    x.paste1.ind <- in.box(x=x.init, box=box.paste1, d=d, boolean=TRUE)
    x.paste1 <- x.init[x.paste1.ind,]
    y.paste1 <- y.init[x.paste1.ind]
    	
    x.paste2.ind <- in.box(x=x.init, box=box.paste2, d=d, boolean=TRUE)
    x.paste2 <- x.init[x.paste2.ind,]
    y.paste2 <- y.init[x.paste2.ind]
	
      
    #while (length(y.paste1) <= length(y) & box.paste1[1,j] >= box.init[1,j])
	while ((length(y.paste1) <= length(y)) & ( box.paste1[1,j] > box.init[1,j] | isTRUE(all.equal(box.paste1[1,j], box.init[1,j]) )))
    {
      box.paste1[1,j] <- box.paste1[1,j] - box.diff*paste.alpha
      x.paste1.ind <- in.box(x=x.init, box=box.paste1, d=d, boolean=TRUE)
      x.paste1 <- x.init[x.paste1.ind,]
      y.paste1 <- y.init[x.paste1.ind]
    }
    
    #while (length(y.paste2) <= length(y) & box.paste2[2,j] <= box.init[2,j])
	 while (length(y.paste2) <= length(y) & ((box.paste2[2,j] < box.init[2,j]) | isTRUE(all.equal(box.paste2[2,j], box.init[2,j]))))
    {
      box.paste2[2,j] <- box.paste2[2,j] + box.diff*paste.alpha
      x.paste2.ind <- in.box(x=x.init, box=box.paste2, d=d, boolean=TRUE)
      x.paste2 <- x.init[x.paste2.ind,]
      y.paste2 <- y.init[x.paste2.ind]
    }

   
    ## y means of pasted boxes
    y.mean.paste[1,j] <- mean(y.paste1)
    y.mean.paste[2,j] <- mean(y.paste2)

    ## mass of pasted boxes
    mass.paste[1,j] <- length(y.paste1)/n
    mass.paste[2,j] <- length(y.paste2)/n
    
    x.paste1.list[[j]] <- x.paste1
    y.paste1.list[[j]] <- y.paste1
    x.paste2.list[[j]] <- x.paste2
    y.paste2.list[[j]] <- y.paste2
    box.paste[1,j] <- box.paste1[1,j]
    box.paste[2,j] <- box.paste2[2,j]
  }

  ## break ties by choosing box with largest mass
  
  y.mean.paste.max <- which(y.mean.paste==max(y.mean.paste, na.rm=TRUE), arr.ind=TRUE)
  
  if (nrow(y.mean.paste.max)>1)
  {
     #cat("more then one box in paste\n")
	 y.mean.paste.max <- cbind(y.mean.paste.max, mass.paste[y.mean.paste.max])
     y.mean.paste.max.ind <- y.mean.paste.max[order(y.mean.paste.max[,3], decreasing=TRUE),][1,1:2]
  }
  else
    y.mean.paste.max.ind <- as.vector(y.mean.paste.max)       
  
  ## paste along dimension j.max
  j.max <- y.mean.paste.max.ind[2]

  ## paste lower 
  if (y.mean.paste.max.ind[1]==1)
  {
     x.new <- x.paste1.list[[j.max]] 
     y.new <- y.paste1.list[[j.max]]
     box.new[1,j.max] <- box.paste[1,j.max]
  }   
  ## paste upper
  else if (y.mean.paste.max.ind[1]==2)
  {
     x.new <- x.paste2.list[[j.max]] 
     y.new <- y.paste2.list[[j.max]]
     box.new[2,j.max] <- box.paste[2,j.max]
  }
  
  mass.new <- length(y.new)/n
  y.mean.new <- mean(y.new)

  if ((y.mean.new > threshold) & (mass.new >= mass.min) & (y.mean.new >= y.mean)
      & (mass.new > mass))
    return(list(x=x.new, y=y.new, y.mean=y.mean.new, box=box.new, mass=mass.new))
 
}




###############################################################################
## Output functions for prim objects
###############################################################################

###############################################################################
## Plot function for PRIM objects
###############################################################################


plot.prim <- function(x, splom=TRUE, ...)
{
  if (ncol(x$x[[1]])==2)
    plotprim.2d(x, ...)
  else if (ncol(x$x[[1]])==3 & !splom)
  {  warning("RGL 3-d plotting temporarily disabled") 
     plotprim.3d(x, ...)
   }
  else if (ncol(x$x[[1]])>3 | (ncol(x$x[[1]])==3 & splom))
    plotprim.nd(x, ...)
  
  invisible()
}

plotprim.2d <- function(x, col, xlim, ylim, xlab, ylab, add=FALSE,
   add.legend=FALSE, cex.legend=1, pos.legend, lwd=1, ...)
{ 
  M <- x$num.hdr.class

  ff <- function(x, d) { return (x[,d]) }

  if (missing(xlim))
    xlim <- range(sapply(x$box, ff, 1))
  if (missing(ylim))
    ylim <- range(sapply(x$box, ff, 2))
  
  x.names <- colnames(x$x[[1]])
  if (is.null(x.names)) x.names <- c("x","y")
  
  if (missing(xlab)) xlab <- x.names[1]
  if (missing(ylab)) ylab <- x.names[2] 
  
  if (!add)
    plot(x$box[[1]], type="n", xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab, ...)
   
  text.legend <- paste("box", 1:M, sep="")
  
  if (missing(pos.legend))
  {  
    pos.legend <- c(xlim[2], ylim[2])
    xlim <- c(xlim[1], xlim[2] + 0.1*abs(xlim[1]-xlim[2]))
  }

  if (missing(col))
  {  
    col <- rep("transparent", M)
    col[which(x$ind==1)] <- "orange"
    col[which(x$ind==-1)] <- "blue"
  }
  
  if (length(col) < M)
    col <- rep(col, length=M)
    
  for (i in M:1)
  {
    ## colour i-th box
    box <- x$box[[i]]
    rect(box[1,1], box[1,2], box[2,1], box[2,2], border=TRUE, col=col[i], lwd=lwd)
  }
  
  if (add.legend)
    legend(pos.legend[1], pos.legend[2], legend=text.legend, fill=col, bty="n",
           cex=cex.legend)
  
  invisible()
}


plotprim.3d <- function(x, color, xlim, ylim, zlim, xlab, ylab, zlab, add.axis=TRUE, size=3, ...)
{
  require(rgl)
  require(misc3d)
  clear3d()
  rgl.bg(color="white")

  M <- x$num.hdr.class
   
  ff <- function(x, d) { return (x[,d]) }
   
  if (missing(xlim))
    xlim <- range(sapply(x$box, ff, 1))
  if (missing(ylim))
    ylim <- range(sapply(x$box, ff, 2))
  if (missing(zlim))
    zlim <- range(sapply(x$box, ff, 3))

  x.names <- colnames(x$x[[1]])
  if (is.null(x.names)) x.names <- c("x","y","z")
  
  if (missing(xlab)) xlab <- x.names[1]
  if (missing(ylab)) ylab <- x.names[2] 
  if (missing(zlab)) zlab <- x.names[3]

  if (add.axis)
  {
    lines3d(xlim[1:2], rep(ylim[1],2), rep(zlim[1],2), size=3, color="black")
    lines3d(rep(xlim[1],2), ylim[1:2], rep(zlim[1],2), size=3, color="black")
    lines3d(rep(xlim[1],2), rep(ylim[1],2), zlim[1:2], size=3, color="black")
  
    texts3d(xlim[2],ylim[1],zlim[1],xlab,size=3,color="black", adj=0)
    texts3d(xlim[1],ylim[2],zlim[1],ylab,size=3,color="black", adj=1)
    texts3d(xlim[1],ylim[1],zlim[2],zlab,size=3,color="black", adj=1)
  }
  
  if (missing(color))
    color <- topo.colors(M)
  if (length(color) < M)
    color <- rep(color, length=M)
  
  for (i in M:1)
  {
    ## colour data in i-th box
    xdata <- x$x[[i]]
    if (is.vector(xdata))
      points3d(xdata[1], xdata[2], xdata[3], color=color[i], size=size)
    else
      points3d(xdata[,1], xdata[,2], xdata[,3], color=color[i], size=size)
  }
  invisible()
}



plotprim.nd  <- function(x, col, xmin, xmax, xlab, ylab, x.pt, m, ...)
{
  M <- x$num.hdr.class
  d <- ncol(x$x)
  if (missing(col))
  {
    ##col <- c(topo.colors(M), "transparent")
    col <- rep("transparent", M)
    col[which(x$ind==1)] <- "orange"
    col[which(x$ind==-1)] <- "blue"
  }
  if (missing(m) & !missing(x.pt)) m <- round(nrow(x.pt)/10)
  if (missing(x.pt))
  {
    x.pt <- numeric()
    for (j in 1:length(x$x))
      x.pt <- rbind(x.pt,x$x[[j]])
    if (missing(m)) m <- max(round(nrow(x.pt)/10), nrow(x$x[[1]]))
    
    x.pt <- x.pt[sample(1:nrow(x.pt), size=m),]
  }

  xprim <- prim.which.box(x.pt, box.seq=x)
  xprim.ord <- order(xprim)
  x.pt <- x.pt[xprim.ord,]
  xprim.col <- col[xprim][xprim.ord]
 
  pairs(x.pt, col=xprim.col,  ...)

  invisible()
}


###############################################################################
## Summary function for PRIM objects
##
## Parameters
## x - prim object
##
## Returns
## matrix, 
## with 2 columns y.mean (mean of y) and mass (box mass)
## i-th row corresponds to i-th box of x
## last row is overall y mean and total mass covered by boxes 
###############################################################################

summary.prim <- function(object, ..., print.box=FALSE)
{
  x <- object
  M <- x$num.class

  if (M>1)
  {  
    summ.mat <- vector()
    for (k in 1:M)
      summ.mat <- rbind(summ.mat, c(x$y.mean[[k]], x$mass[[k]], x$ind[k]))
    
    tot <- c(sum(summ.mat[,1]*summ.mat[,2])/sum(summ.mat[,2]), sum(summ.mat[,2]), NA)
    summ.mat <- rbind(summ.mat, tot)
    
    rownames(summ.mat) <- c(paste("box", 1:(nrow(summ.mat)-1), sep=""), "overall")
    colnames(summ.mat) <- c("box-mean", "box-mass", "threshold.type")
    
    if (x$num.hdr.class < x$num.class)
      for (k in (x$num.hdr.class+1):x$num.class)
      rownames(summ.mat)[k] <- paste(rownames(summ.mat)[k], "*",sep="")
  }
  else
  {
    summ.mat <- c(x$y.mean[[1]], x$mass[[1]])
    tot <- c(x$y.mean[[1]], x$mass[[1]])
    summ.mat <- rbind(summ.mat, tot)
    rownames(summ.mat) <- c(paste("box", 1:(nrow(summ.mat)-1), sep=""), "overall")
    colnames(summ.mat) <- c("box-mean", "box-mass")
  }
  
  print(summ.mat)
  cat("\n")
  
  if (print.box)
  { 
    for (k in 1:M)
    {
      cat(paste("Box limits for box", k, "\n", sep=""))
       box.summ <- x$box[[k]]
       rownames(box.summ) <- c("min", "max")
       print(box.summ)
      cat("\n")
    }
  }
}