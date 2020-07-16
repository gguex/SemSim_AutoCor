# calculs auto-corrélation textuelle avec "sganarelle" (données préparées par Aris)

# on considère divers types de distances D entre mots, ainsi que différentes matrices d'échange E(l) ("l-sized neighborhoods")

setwd("/Documents/SimDiversity/SemSim_AutoCor/code_R/textual_autocorrelation_old")

  
distances=read.csv(file="distances.txt",sep = "\t",header = TRUE)

pos1=distances$position1
pos2=distances$position2
discr=distances$discrete
wdiscr=distances$weighted_discrete
chiSq=distances$chi_square


sexe=read.csv(file="sganarelle_sexe.txt",sep = "\t",header = TRUE)
pos=sexe$X__context__
femme=sexe$femme

docterme=read.csv(file="sganarelle_doc_terme.txt",sep = "\t",header = TRUE)
docterme=as.matrix(docterme)
Encoding(docterme) <- "latin1"


context=as.numeric(docterme[,3])
count=as.numeric(docterme[,4])

longueur<-aggregate(count,list(context=context),sum)[,2]
n=length(longueur)

 
#Extraction des matrices de distances
#longueurs de mots
Dlong=matrix(0,n,n)
for (i in 1:n) {
	for (j in 1:n) {
	Dlong[i,j]=(longueur[i]-longueur[j])^2
	}}



#sexe des personnes
Dsex=matrix(0,n,n)
for (i in 1:n) {
	for (j in 1:n) {
	Dsex[i,j]=(femme[i]-femme[j])^2
	}}

#chi-carre
dim(chiSq)=c(n,n)
Dchi2=chiSq

#discrete === c'est la même que Dsex! 
dim(discr)=c(n,n)
Ddiscr=discr

#discrete pondérée
dim(wdiscr)=c(n,n)
Dwdiscr=wdiscr

# choix de la distance 
D=Dwdiscr
# D=Dlong
# etc.

# code R pour creer les "l-sized neighborhoods"
 

E<-function(l){
	E<-rep(0,n*n)
	L<-rep(0,n*n)
	dim(L)=c(n,n)
	dim(E)=c(n,n)
	for (i in 1:n){
		for (j in 1:n){
			L[i,j]<-as.numeric(abs(i-j)<=l)*(1-as.numeric(j==i))
		}
	}	
	E<-L/sum(L)
	E
} 



#inertie, inertie locale et difference relative, de l=1 a l=K

#initialisation
K<-n
diffrel<-c()
diffrel_max<-c()
diffrel_min<-c()
esp = c()
alpha = 0.05

for (l in 1:K){
	f<-rowSums(E(l))
	Delta<-0.5*(t(f)%*%D%*%f)
	DeltaLoc<-0.5*sum(E(l)*D)
	diffrel[l]<-(Delta-DeltaLoc)/Delta
	
	#Calcul de la moyenne et l'espérance de l'index
	#d'autocorrélation
	W = diag(1/f)%*%E(l)
	esp[l] = (sum(diag(W))-1)/(n-1)
	var = (2/(n^2-1))*(sum(diag(W%*%W))-1-(((sum(diag(W))-1)^2)/(n-1)))
	
	#Calcul du delta max et min pour alpha = 0.05
	diffrel_max[l] = (sqrt(var)*qnorm(1-(alpha/2)))+esp[l]
	diffrel_min[l] = (sqrt(var)*qnorm(alpha/2))+esp[l]

} 


plot(diffrel[1:50], type = "b",  ylim = c(diffrel[1],diffrel[2]), xlab =expression("neighbourhood size "*r), ylab =expression("autocorrelation index "*delta),mgp=c(2, 0.8, 0))
lines(esp[1:50])
lines(diffrel_max[1:50], lty=2)
lines(diffrel_min[1:50], lty=2)
