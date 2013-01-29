library('prim')
source('C:/workspace/EMA-workbench/src/analysis/primCode/R/prim.R')
source('C:/workspace/EMA-workbench/src/analysis/primCode/R/boxfun.R')
peel.alpha <- 0.05
paste.alpha <- 0.01 
mass.min <- 0.05
threshold <- -1.1
pasting <- TRUE
verbose <- TRUE

test.peel.one <- function()
{
	x <- read.table('C:/workspace/EMA workbench/src/analysis/primCode/data', sep=" ")[,1:2]
	y <- read.table('C:/workspace/EMA workbench/src/analysis/primCode/data', sep=" ")[,3]
	
	peel.alpha <- 0.05
	paste.alpha <- 0.01 
	box.init <- apply(x, 2, range) 
	box.diff <- box.init[2,] - box.init[1,]
	box.init[1,] <- box.init[1,] - 10*paste.alpha*box.diff
	box.init[2,] <- box.init[2,] + 10*paste.alpha*box.diff

	threshold <- -1.1
	d <- ncol(x)
	n <- nrow(x)	
	mass.min <- 0.05
	type <- 8

	box <- peel.one(x, y, box.init, peel.alpha, mass.min, threshold, d, n, type)
	
	cat("mass: ", box$mass, "\n", sep="") 
	print(box$box)
	cat("mean: ", box$y.mean, "\n",sep="")
}

test.find.box <- function()
{
	x <- read.table('C:/workspace/EMA workbench/src/analysis/primCode/data', sep=" ")[,1:2]
	y <- read.table('C:/workspace/EMA workbench/src/analysis/primCode/data', sep=" ")[,3]
	
	peel.alpha <- 0.05
	paste.alpha <- 0.01 
	box.init <- apply(x, 2, range) 
	box.diff <- box.init[2,] - box.init[1,]
	box.init[1,] <- box.init[1,] - 10*paste.alpha*box.diff
	box.init[2,] <- box.init[2,] + 10*paste.alpha*box.diff

	threshold <- -1.1
	d <- ncol(x)
	n <- nrow(x)	
	mass.min <- 0.05

	pasting <- FALSE
	verbose <- TRUE

	box <- find.box(x, y, box.init, peel.alpha, paste.alpha, mass.min, threshold, d, n, pasting, verbose)
	cat("mass: ", box$mass, "\n", sep="") 
	print(box$box)
	cat("mean: ", box$y.mean, "\n",sep="")
}

test.prim.one <- function()
{
	x <- read.table('C:/workspace/EMA workbench/src/analysis/prim/quasiflow x.txt', sep="\t")[1:200,]
	y <- read.table('C:/workspace/EMA workbench/src/analysis/prim/quasiflow y.txt', sep="\t")[1:200,]
	
	peel.alpha <- 0.05
	paste.alpha <- 0.01
	box.init <- NULL
	mass.min <- 0.05
	threshold <- 0.1
	pasting <- FALSE
	verbose <- TRUE
	threshold.type <- 1
	y <- y*threshold.type
	
	boxes <- prim.one(x, y, box.init, peel.alpha, paste.alpha, mass.min, threshold, pasting, threshold.type,verbose)
}

test.box <- function()
{
    x <- read.table('C:/workspace/EMA-workbench/src/analysis/primCode/quasiflow x.txt', sep="\t")
    y <- read.table('C:/workspace/EMA-workbench/src/analysis/primCode/quasiflow y.txt', sep="\t")
	
	x = x[1:1000,1:2]
	y = y[1:1000,]
	# Error: object 'x' not found
	
    boxes <- prim.box(x, y, verbose=TRUE, threshold.type=1)
}
