classdef HumanController < handle
    %CONTROLLER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        num
        den
        
        e
        e_1
        e_2
        u
        u_1
        u_2
        
        gain
        sat
        
        pos
        error
        controlLaw
        
        joint
        
        signals
        info
        
        upT
    end
    
    methods
        function obj = HumanController(jointComponent, P, D, I, N ,dt, upT, gain, sat)
            %CONTROL    LER Construct an instance of this class
            %   Detailed explanation goes here
            obj.sat = sat;
            obj.gain = gain;
            obj.upT = upT;
            obj.joint = jointComponent;
            
            H_P = tf(P,1);
            H_I = tf(I,[1,0]);
            H_D = tf([D*N,0],[1,N]);
            H=H_P+H_I+H_D;
            
            Hd = c2d(H,dt,'zoh');
            
            obj.num = Hd.Numerator{1,1};
            obj.den = Hd.Denominator{1,1};
            
%             step(H,Hd)
            obj.e_1 = 0;
            obj.e_2 = 0;
            obj.u_1 = 0;
            obj.u_2 = 0;
            
            obj.signals = [];
            obj.info = [0,P,D,I];
            
        end
        
        function f = update(obj,state,ref,t)
            
            % current position:
            p = obj.joint.getCoordinate().getValue(state);
            % error
            obj.e = ref - p;
            
            obj.u =  (1/obj.den(1))*(obj.gain *(obj.num(1)*obj.e+obj.num(2)*obj.e_1+obj.num(3)*obj.e_2)-obj.den(2)*obj.u_1-obj.den(3)*obj.u_2);
            
            obj.e_2 = obj.e_1;
            obj.e_1 = obj.e;
            
            obj.u_2 = obj.u_1;
            obj.u_1 = obj.u;
            
            f= obj.gain*obj.u;

            
            if f > obj.sat
                f = obj.sat;
            elseif f < -obj.sat
                f = -obj.sat;
            end
            
            
            if mod(t,obj.upT)==0
                obj.signals = [obj.signals;[t, ref, obj.e, obj.u,f]];
            end
        end
            
        function updateController(obj, newP, newD, newI, N, dt, t)
            H_P = tf(newP,1);
            H_I = tf(newI,[1,0]);
            H_D = tf([newD*N,0],[1,N]);
            H=H_P+H_I+H_D;
            
            Hd = c2d(H,dt,'zoh');
            
            obj.num = Hd.Numerator{1,1};
            obj.den = Hd.Denominator{1,1};
            obj.info = [obj.info;[t, newP, newD, newI]];
%             step(H,Hd)
%             obj.e_1 = 0;
%             obj.e_2 = 0;
%             obj.u_1 = 0;
%             obj.u_2 = 0;
        end
    end
end

