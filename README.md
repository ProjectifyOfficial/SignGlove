# SignGlove 

## About

In the context of ECESCON 9 Hackathon, we have developed as a team a low-cost gesture glove for interpretation of sign language. This glove enables hearing impaired users to capture sign language notices (at a prominent stage with the requirements of the competition) through a cross-platform mobile and computer application that allows them to be recognized, recorded and spoken, translate nods into spoken text, facilitating their communication with other people.

## Team

The team comprises of
1. Michalis Meggisoglou
2. Evangelos Manolis
3. Marios Papachristou

---

# Implementation of the Project

## Material

The creation of the gauntlet gloves was mainly done with the use of simple objects, namely a bicycle glove, various common objects (such as finger rings), an Arduino Uno and flex sensors, which also constituted the most important part of the material for the purpose of the project. These sensors, which act as variable resistors, were added to each finger and held back using rivets and dashes to ensure a sufficient stability. Between their ends, a potential difference was created that we could scan through the analog inputs of Arduino and send it to the smartphone via a serial connection.


## Software

The piece of our software project consisted of creating a cross-platform smartphone application to identify gestures, record them on the screen, and text-to-speech -Speech (TTS) software. The writing of the source code was done in Python language, since its ease of use and application development speed fit perfectly into the tight time frame of the event. Also, our requirement to create a cross-platform application has contributed significantly to the choice of the particular language for writing the code. As regards the development of the Graphical Interface Environment (GUI), the Kivy library was used, which easily allowed us to create the graphical environment necessary for the group's needs. Finally, the software runs on a Python interpreter, of course, but it is not possible to block an APK to transfer it to other Android devices that do not have a language interpreter installed.

In particular, our application has the ability to "learn" learning to learn nodes that are to be recognized in the recognition phase. In this process, for each given symbol of the alphabet, based on the data from the sensors, specific number intervals are used which are then used in the recognition phase. More specifically, in this final phase of identification of the nodes, for each symbol the arithmetic output of each sensor is controlled and the set of the sensors belonging to the respective predetermined intervals is detected. If this set contains sufficient data, then the algorithm considers that the corresponding alphabetical symbol is activated and then prints it on the screen. The simplicity of this algorithm is due, on the one hand, to tight deadlines, on the other hand to complexity (mainly computational) involving more complex algorithms such as machine learning and calibration procedures to achieve "more natural" (see Future Extensions). In addition, the integration of the smartphone into the user's bar allowed us to add extra moves (such as saying the word with 90 degrees or, in the future, changing to "number input mode" with a nod) .

Finally, our implementation also has a client to speak the words from the smartphone / computer of the interlocutor via Bluetooth, for the full autonomy of the interacting users.

# Bill of materials

The gesture glove is made of relatively simple materials (bicycle glove, thimbles), flex sensors and an Arduino Uno, which means that the present work is intended to be a low cost device. It should also be noted that all the structures used (hardware and software) are open / open / free / open source software. Thus, the total cost was around 40EUR.
Arduino Uno: 20 EUR
Pressure Sensors: 11 EUR
Glove and other items: <8 EUR
Total: <40 EUR

# Future Extensions

## Algorithmic Improvement Using Machine Engineering Algorithms

Mutual detection can be achieved and improved by using engineering learning algorithms such as coiled neural networks (Back Propagation) and Probabilistic k-Nearest Neighbours with weighted metrics [5,6]. Initially the model will be trained by the user himself, and then the detection of the veins is done by matching them to classes based on the classification algorithm. Finally, it is worth noting that the team had implemented during the competition an engineering learning algorithm, which was not used due to the lack of training datasets available at that time as well as due to time constraints that did not allow the complete training of the algorithm.

## Graphical Interface Improvement

The primary concern is the construction of an environmentally friendly end-user interface (GUI) to increase and improve the usability and functionality of the project. We clarify that the present interface environment is at the forefront and mainly serves the needs of the team, as it leaves out important data regarding its stability of execution, its aesthetics and its ease of use. However, with a careful approach (using aesthetically more beautiful bark) and maintaining its existing layout as it is, we believe that the desired result can be achieved.

## Improvement of Material

Improving the work from a material standpoint is to create a more stable prototype to eliminate recurring calibrations, the use of appropriate materials, and rotation of the design to make it easy to use and enjoyable. Also, consideration is being given to adding more low-cost sensors to offer increased functionality at relatively low cost.

## Other improvements: Integration into the Internet of Things and interfacing with other wearables

Undoubtedly, the use of a hearing glove by hearing impaired people can be a way of communicating with the rest of the world, as well as data logging and data mining when using these applications is an enormous addition to the Internet of Things, defined as [8,9] as the network of interactive devices that exchange and collect data. As Mari Carmen Domingo says (2011, [7]):

> "[...] On the other hand, sensory or mentally handicapped children can use interactive play and learning IoT environments ('at school' scenario) to experience a richer learning experience (cognitive skill development), have more opportunities for language acquisition linguistic skills, become better at interacting with others (social skill development) and thus obtain higher self-esteem (Hengeveld et al., 2009). In addition, these interactive IoT systems adapt to their learning rhythm. For example, deaf children often need additional exposure to the American Sign Language, because most of their parents are hearing and not fluent in this language (Parton et al., 2009). Now they can take this interactive IoT system with them from school to home and repeat the most difficult vocabulary until they understand and memorize it. Learning is less difficult with this innovative and friendly system (it turns into a game) and learning barriers are reduced. This is especially important, since there is a strong correlation between early sign language exposure for deaf children and later academic achievement (Parton et al., 2009). [...] "

Therefore, the use and incorporation of a nose gauntlet into a wider set of networked devices would allow for more accurate identification of nudes (due to the plethora of exercise data) and, on the other hand, for the recording of nudes and possibly the very "reason" people who use it to investigate motives in their language and linguistic pattern analysis and behavioral analysis.

Additionally, other smart accessories such as wearables can be used against the smartphone. For example, using a smartwatch may well reduce the "size" of the overall build, contributing positively to the usability and functionality of the entire project. Thus, the interconnection of the glove with a variety of other devices imparts increased modularity and hybridity to the set as well as makes it easier to interface with third party devices, either commercial or peer - to - peer - purpose [4].

