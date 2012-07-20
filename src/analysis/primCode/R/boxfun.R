

####################################################################
### Find points of x that are in a single box
###
### Parameters
### x - data matrix
### ranges - matrix of min and max values which define a box 
### d - dimension of data
###
### Returns
### Data points which lie within the box
####################################################################

in.box <- function(x, box, d, boolean=FALSE)
{
  x.box.ind <- rep(TRUE, nrow(x)) 
  for (i in 1:d)
     x.box.ind <- x.box.ind & (box[1,i] <= x[,i]) & (x[,i] <= box[2,i])

  if (boolean)
    return(x.box.ind)
  else  
    return(x[x.box.ind,])
}


###############################################################################
## Allocates data x according to a sequence of boxes
##
## Parameters
## x - data matrix
## y - response values
## box.seq - list of boxes (output from prim is OK)
##
## Returns
## List with k fields, one for each box
## each field in turn is a list with fields
## x - data in box
## (y - corresponding response values)
## (y.mean - mean of y)
## box - box limits
## box.mass - box mass 
##
## NB: if y is missing from the call, then $y and $y.mean aren't computed
###############################################################################


in.box.seq <- function(x, y, box.seq)
{
  m <- box.seq$num.class
  d <- ncol(x)
  n <- nrow(x)
  
  x.ind <- rep(TRUE, n)
  xy.list <- list()

  for (k in 1:m)
  {
    x.ind.curr <- x.ind    
    box.curr <- box.seq$box[[k]]
    
    for (j in 1:d)
      x.ind.curr <- x.ind.curr & (x[,j]>= box.curr[1,j]) & (x[,j] <= box.curr[2,j])
    
    x.curr <- x[x.ind.curr & x.ind,]
    box.mass.curr <- sum(x.ind.curr)/n
  
    xy.list$x[[k]] <- x.curr
    if (!missing(y))
    {
      y.curr <-  y[x.ind.curr & x.ind]
      y.mean.curr <- mean(y.curr)
      xy.list$y[[k]] <- y.curr
      xy.list$y.mean[[k]] <- y.mean.curr
    }
    xy.list$box[[k]] <- box.curr
    xy.list$mass[[k]] <- box.mass.curr
   
    ## exclude those in in current box (x.ind.curr) for the next iteration
    x.ind <- x.ind & !x.ind.curr
  }
  return (xy.list)
}


###############################################################################
## Returns the box number which the data points belong in
##
## Parameters
##
## x - data matrix
## box.seq - list of boxes
##
## Returns
##
## Vector of box numbers
###############################################################################

prim.which.box <- function(x, box.seq)
{
  if (is.vector(x)) x <- t(as.matrix(x))

  m <- box.seq$num.class
  d <- ncol(x)
  n <- nrow(x)

  x.ind <- rep(TRUE, n)
  x.which.box <- rep(0,n)

  for (k in 1:m)
  {
    x.ind.curr <- x.ind    
    box.curr <- box.seq$box[[k]]
    
    for (j in 1:d)
      x.ind.curr <- x.ind.curr & (x[,j]>= box.curr[1,j]) & (x[,j] <= box.curr[2,j])
    
    x.which.box[x.ind.curr & x.ind] <- k  
   
    ## exclude those in current box (x.ind.curr) for the next iteration
    x.ind <- x.ind & !x.ind.curr
  }
  
  return (x.which.box)
}


###############################################################################
## Count the number of data points x which fall into a sequence of boxes
##
## Parameters
## x - data matrix
## box.seq - sequence of boxes (prim object)
##
## Returns
## Vector of counts, i-th count corr. to i-th box
###############################################################################

counts.box <- function(x, box.seq)
{
  m <- box.seq$num.class
  x.counts <- rep(0, m)
  x.class <- prim.which.box(x, box.seq)
 
  for (k in 1:m)
    x.counts[k] <- sum(x.class==k)
 
  return(x.counts)
}


###############################################################################
## Hypervolume of hyperbox
##
## Parameters
## box - matrix of box limits
##
## Returns
## hypervolume of a hyperbox
###############################################################################

vol.box <- function(box)
{
  return(prod(abs(box[2,] - box[1,])))
}




####################################################################
## Decide whether two box sequences overlap each other
##
## Input
## box.seq1 - first box sequence
## box.seq2 - second box sequence
##
## Returns
## TRUE if they overlap, FALSE o/w
####################################################################

overlap.box.seq <-function(box.seq1, box.seq2, rel.tol=0.01)
{
  M1 <- box.seq1$num.hdr.class
  M2 <- box.seq2$num.hdr.class
  d <- ncol(box.seq1$box[[1]])

  overlap.mat <- matrix(FALSE, nrow=M1, ncol=M2)
  for (i in 1:M1)
  {
    box1 <- box.seq1$box[[i]]
    for (j in 1:M2)
    {  
      box2 <- box.seq2$box[[j]]
      overlap.mat[i,j] <- overlap.box(box1, box2, rel.tol=rel.tol)
    }
  }

  return(overlap.mat)
}

####################################################################
## Decide whether two boxes overlap each other
##
## Input
## box1 - first box 
## box2 - second box
##
## Returns
## TRUE if they overlap, FALSE o/w
####################################################################


overlap.box <-function(box1, box2, rel.tol=0.01)
{
  d <- ncol(box1)

  overlap <- TRUE

  box1.tol <- box1
  box1.range <- abs(apply(box1, 2, diff))
  box1.tol[1,] <- box1.tol[1,] + rel.tol*box1.range
  box1.tol[2,] <- box1.tol[2,] - rel.tol*box1.range
  
  box2.tol <- box2
  box2.range <- abs(apply(box2, 2, diff))
  box2.tol[1,] <- box2.tol[1,] + rel.tol*box2.range
  box2.tol[2,] <- box2.tol[2,] - rel.tol*box2.range
  
  for (k in 1:d)
    overlap <- overlap & (((box1.tol[1,k] <= box2.tol[1,k]) & (box2.tol[1,k] <= box1.tol[2,k]))
                          | ((box1.tol[1,k] <= box2.tol[2,k]) & (box2.tol[2,k] <= box1.tol[2,k]))
                          | ((box2.tol[1,k] <= box1.tol[1,k]) & (box1.tol[1,k] <= box2.tol[2,k]))
                          | ((box2.tol[1,k] <= box1.tol[2,k]) & (box1.tol[2,k] <= box2.tol[2,k])))
  
  return(overlap)
}
