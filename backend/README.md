In our framework, everything is a class. The notion of using Object-Oriented
Programming stems from the amount of information the different entities need
to share with each other. Every generation/optimization step, the algorithms
core engines need to make several decisions based on the current and past
statuses, as well as the feedback from the environment. Using OOP facilitates
this information-sharing.

In addition, OOP removes duplicate code, and maintains clean definitions of
entities that can be re-used in different algorithms. Thus, again, almost
all the entities are classes, and then there are action scripts to act on objects
of those classes.

Furthermore, we like to keep the base definitions concise and extend the classes
in the respective implementations as needed. For example, in the Environment
base class, there are only a few methods. Those are expected to be shared amongst
all environments. This is a CRITICAL unification/standardization of interfaces.
The idea is to create a seamless plug-and-play behavior between the algorithms
and the environments.

Finally, the base class implementations themselves are typically "barebone". That
is, we attempt to leave the rigorous definitions of methods and such to the
extended/child classes rather than the base class. Think of base class as a
"template".
