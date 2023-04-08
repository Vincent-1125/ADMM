
load('Lobby_SwitchLight_ALL_1000_2545_1546_128_160_R')
M = images(:,351:430);
rho = 0.014;
[L, S] = MYRPCASolver(M, rho, 1e-4, 220);

function [L, S] = MYRPCASolver(M, rho, e, k)
tic
p=128; q=160; v=80; 
m=20480; n=80;

niter=0;
maxiter = k;
t = 0.035;

Y = sparse(m,n);
X = M - Y;
A = rand(m,n);
ek = 1;

while niter <= maxiter && ek > e

    Xnew = argminx(Y, M, A, t);
    Ynew = argminy(Xnew, Y, M, A, t, rho);
    Anew = A - t * (Xnew + Ynew - M);

    ek  = norm(Xnew + Ynew - M);

    X = Xnew;
    Y = Ynew;
    A = Anew;

    niter = niter+1;
    disp(niter);    disp(ek)
    
end

L = X;
S = Y;
Show(M, L, S, p, q, v);
toc

end


function Show(M, L, S, p, q, v)

T=zeros(p,q,v);
B=zeros(p,q,v);
F=zeros(p,q,v);
for i=1:v
    T(:,:,i)=reshape(M(:,i),p,q);
    B(:,:,i)=reshape(L(:,i),p,q);
    F(:,:,i)=reshape(S(:,i),p,q);
end
subplot(331);   imshow(uint8(T(:,:,5)));
subplot(334);   imshow(uint8(T(:,:,15)));
subplot(337);   imshow(uint8(T(:,:,35)));
subplot(332);   imshow(uint8(B(:,:,5)));
subplot(335);   imshow(uint8(B(:,:,15)));
subplot(338);   imshow(uint8(B(:,:,35)));
subplot(333);   imshow(uint8(F(:,:,5)));
subplot(336);   imshow(uint8(F(:,:,15)));
subplot(339);   imshow(uint8(F(:,:,35)));

end


function Xnew = argminx(Y, M, A, t)

Y1 = -Y+M+A/t;
[U,S,V] = svd(Y1,"econ");
[r,c] = size(S);
Sv = ones(r,c);
for i = 1:r
    for k = 1:c
        Sv(i,k) = max(0, S(i,k)-1);
    end
end
Xnew = U*Sv*V';

end


function Ynew = argminy(X, Y, M, A, t, rho)

b = M + A/t - X;
Y = Y-(Y-b)/t;
y = abs(Y)-rho/t;
y((y<0)) = 0;
Ynew = sign(Y).*y;

end
