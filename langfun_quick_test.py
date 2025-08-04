import langfun as lf
import pyglove as pg

r = lf.query('Who is Larry Page', lm=lf.llms.Gpt4o())
print(r)




image = lf.Image.from_uri(
    'https://i.imgur.com/KhGVVdF.jpeg'
)

class ImageDescription(pg.Object):
  description: str
  objects: list[str]

r = lf.query(prompt='{{image}}', schema=ImageDescription, image=image, lm=lf.llms.Gpt4o())
print(r)


code = lf.query(prompt='plot y = x ** 2', schema=lf.PythonCode, lm=lf.llms.Gpt4o())

# Execute the generated code.
#code(sandbox=False, autofix_lm=lf.llms.Gpt4o())

# View the source code
print(code.source)