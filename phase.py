from coldtype import *
from coldtype.fx.skia import phototype, potrace

aspect = 16/9
width = 1920
height = width / aspect

fonts = [
    "/Users/matter/Downloads/The Chaffy Otf - Envato Elements/TheChaffy-Bold.otf",
    "/Users/matter/Downloads/The Chaffy Otf - Envato Elements/TheChaffy-Italic.otf",
    "/Users/matter/Downloads/The Chaffy Otf - Envato Elements/TheChaffy-Regular.otf",
]
fn = fonts[0]

fn = "/Users/matter/Downloads/cobya-modern-variable-font-2023-11-27-05-16-13-utc/Variable/CobyaVariableGX.ttf"

@animation((width, height), tl=(20, 140/60))
def scratch(f:Frame):
    left, right = f.a.r.divide(0.5, "W")

    l = (
        StSt(str(f.i%4+1), fn, 72, wght=0.7)
        .f(1)
        .align(left)
        .scaleToRect(left.inset(100))
    )

    r = (
        StSt(str(f.i%5+1), fn, 72, wght=0.7)
        .f(1)
        .align(right)
        .scaleToRect(right.inset(100))
    )

    comp = P()
    comp += P(
        P( 
          P().rect(f.a.r).f(0),
          P(l,r).fssw(1,1,70)
        )
        .ch(phototype(f.a.r,5.9,151,20))
        .ch(potrace(f.a.r))
        .f(hsl(0.9, 0.8, 0.7))
        ,
    )

    comp.insert(0, P().rect(f.a.r).f(hsl(0.5, 0.7, 0.8)))

    return comp
