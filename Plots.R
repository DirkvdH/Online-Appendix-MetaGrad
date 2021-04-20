# plots was er een figuur

results <- read.csv("results.csv")

RC <- results
RC[5:9]
minlim <- ceiling(min(RC[c('GDt','MGCo','MGF11','AdaGrad','MGFull')], na.rm = T))
maxlim <- ceiling(max(RC[c('MGCo','MGF11','AdaGrad','MGFull')], na.rm = T))
GDtminlim <- ceiling(min(RC['GDt'], na.rm = T))
GDtmaxlim <- ceiling(max(RC['GDt'], na.rm = T))
plotchar = 15:18
names(plotchar) <- c("absolute", "hinge", "logistic", "squared")
plotcol = c("black", "red", "forestgreen", "blue")
names(plotcol) <- c("absolute", "hinge", "logistic", "squared")

legendtab <- unique(data.frame(RC$loss, 
                    plotcol[RC$loss], 
                    plotchar[RC$loss]))

resetpar <- par()

pdf(file = "GDvsMG.pdf", width = 9, height = 9)

op <- par(mfrow = c(2,2),
          oma = c(5,4,0,0) + 0.2,
          mar = c(0,0,1,1) + 0.1)

# all in one
plot(RC[c('GDt', "MGCo")], xlim = c(GDtminlim, GDtmaxlim), ylim = c(minlim, maxlim), 
     col = plotcol[RC$loss], cex = 1.5, 
     pch = plotchar[RC$loss], xlab = "", 
     ylab = "",log="xy",axes=F)
myTicks = 10^(0:14)
#axis(1, at = myTicks, labels = parse(text=paste0("'10'^",log10(myTicks))))
axis(1, at = myTicks, labels = F)
axis(2, at = myTicks, labels = parse(text=paste0("'10'^",log10(myTicks))))
#axis(side = 1, labels = F, at = (0:14)[(0:14) %% 2 == 1])
#axis(side = 2, labels = round(exp((0:14)[(0:14) %% 2 == 1]), 0), at = (0:14)[(0:14) %% 2 == 1])
box(which = "plot", bty = "l")

abline(0, 1, col = "black", lwd = 2)
#legend("bottomright", legend = legendtab[, 1], text.col = legendtab[, 2], 
#       pch = as.numeric(legendtab[, 3]), col = legendtab[, 2])
legend("topleft", legend = "MetaGrad Coordinate", cex = 1.5, bty = "n")


plot(RC[c('GDt', "MGF11")], xlim = c(GDtminlim, GDtmaxlim), ylim = c(minlim, maxlim), 
     col = plotcol[RC$loss], cex = 1.5, 
     pch = plotchar[RC$loss], xlab = "", 
     ylab = "", log="xy",axes = F)
axis(1, at = myTicks, labels = F)
axis(2, at = myTicks, labels = F)
#axis(side = 1, labels = F, at = (0:14)[(0:14) %% 2 == 1])
#axis(side = 2, labels = F, at = (0:14)[(0:14) %% 2 == 1])
box(which = "plot", bty = "l")
abline(0, 1, col = "black", lwd = 2)
#legend("bottomright", legend = legendtab[, 1], text.col = legendtab[, 2], 
#       pch = as.numeric(legendtab[, 3]), col = legendtab[, 2])
legend("topleft", legend = "MetaGrad F11", cex = 1.5, bty = "n")


plot(RC[c('GDt', "AdaGrad")], xlim = c(GDtminlim, GDtmaxlim), ylim = c(minlim, maxlim), 
     col = plotcol[RC$loss], cex = 1.5, 
     pch = plotchar[RC$loss], xlab = "", 
     ylab = "", log="xy",axes = F)
axis(1, at = myTicks, labels = parse(text=paste0("'10'^",log10(myTicks))))
axis(2, at = myTicks, labels = parse(text=paste0("'10'^",log10(myTicks))))
#axis(side = 1, labels = round(exp((0:14)[(0:14) %% 2 == 1]), 0), at = (0:14)[(0:14) %% 2 == 1])
#axis(side = 2, labels = round(exp((0:14)[(0:14) %% 2 == 1]), 0), at = (0:14)[(0:14) %% 2 == 1])
box(which = "plot", bty = "l")
abline(0, 1, col = "black", lwd = 2)
#legend("bottomright", legend = legendtab[, 1], text.col = legendtab[, 2], 
#       pch = as.numeric(legendtab[, 3]), col = legendtab[, 2])
legend("topleft", legend = "AdaGrad", cex = 1.5, bty = "n")


plot(RC[c('GDt', "MGFull")], xlim = c(GDtminlim, GDtmaxlim), ylim = c(minlim, maxlim), 
     col = plotcol[RC$loss], cex = 1.5, 
     pch = plotchar[RC$loss], xlab = "", 
     ylab = "", log="xy",axes = F)
axis(1, at = myTicks, labels = parse(text=paste0("'10'^",log10(myTicks))))
axis(2, at = myTicks, F)
#axis(side = 1, labels = round(exp((0:14)[(0:14) %% 2 == 1]), 0), at = (0:14)[(0:14) %% 2 == 1])
#axis(side = 2, labels = F, at = (0:14)[(0:14) %% 2 == 1])
box(which = "plot", bty = "l")
abline(0, 1, col = "black", lwd = 2)
legend("bottomright", legend = legendtab[, 1], text.col = legendtab[, 2], 
       pch = legendtab[, 3], col = legendtab[, 2], cex = 1.5)
legend("topleft", legend = "MetaGrad Full", cex = 1.5, bty = "n")

title(xlab = "Regret of Online Gradient Descent",
      ylab = "Regret",
      outer = TRUE, line = 3, cex.lab = 1.5)
dev.off()


par <- resetpar





# all in one
pdf(file = "GDvsADA.pdf", width = 9, height = 9)

plot(log(RC[c('GDt', "AdaGrad")]), xlim = c(0, maxlim), ylim = c(0, maxlim), 
     col = plotcol[RC$loss], cex = 1.5, 
     pch = plotchar[RC$loss], xlab = "Log(Regret) Online Gradient Descent", 
     ylab = "Log(Regret) AdaGrad", axes = T, main = "GDt vs AdaGrad")
abline(0, 1, col = "black", lwd = 2)

dev.off()
